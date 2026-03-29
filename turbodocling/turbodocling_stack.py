from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    CfnOutput,
    aws_s3 as s3,
    aws_sqs as sqs,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_autoscaling as autoscaling,
)
from constructs import Construct


class TurboStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        stage: str = "dev",
        pdfium_variant: str = "upstream",
        pypdfium2_wheel_url: str = "",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.stage = stage
        self.pdfium_variant = pdfium_variant.lower()
        self.pypdfium2_wheel_url = pypdfium2_wheel_url

        if self.pdfium_variant not in {"upstream", "experimental"}:
            raise ValueError(
                f"Unsupported pdfium_variant={self.pdfium_variant!r}; expected 'upstream' or 'experimental'."
            )
        if self.pdfium_variant == "experimental" and not self.pypdfium2_wheel_url:
            raise ValueError(
                "pypdfium2_wheel_url is required when pdfium_variant='experimental'."
            )

        # S3 bucket for documents (input PDFs, intermediate batches, output markdown/JSON)
        documents_bucket = s3.Bucket(
            self, "DocumentsBucket",
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )

        # SQS dead letter queue for failed GPU jobs
        gpu_dlq = sqs.Queue(
            self, "GpuProcessingDLQ",
            retention_period=Duration.days(14),
        )

        # SQS queue for GPU processing jobs
        gpu_queue = sqs.Queue(
            self, "GpuProcessingQueue",
            visibility_timeout=Duration.seconds(3600),
            retention_period=Duration.days(1),
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=3,
                queue=gpu_dlq,
            ),
        )

        docker_build_args = {
            "PDFIUM_VARIANT": self.pdfium_variant,
            "PYPDFIUM2_WHEEL_URL": self.pypdfium2_wheel_url,
        }

        # Lambda: PDF page preprocessing (SnapStart, ARM64)
        pdf_processor = _lambda.Function(
            self, "ProcessPageBatch",
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="handler.lambda_handler",
            architecture=_lambda.Architecture.ARM_64,
            timeout=Duration.seconds(30),
            memory_size=1769,
            environment={
                "STAGE": self.stage,
                "DOCUMENTS_BUCKET": documents_bucket.bucket_name,
                "PDFIUM_VARIANT": self.pdfium_variant,
            },
            code=_lambda.Code.from_docker_build(
                ".",
                file="lambdas/process_page_batch/Dockerfile.lambda",
                platform="linux/arm64",
                build_args=docker_build_args,
            ),
            snap_start=_lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
            log_retention=logs.RetentionDays.ONE_WEEK,
        )

        documents_bucket.grant_read_write(pdf_processor)

        # Force version creation for SnapStart
        lambda_version = pdf_processor.current_version

        # Step Function definition
        # Compute optimal batch_size: ceil(total_pages / 40)
        # Step Functions lacks division, so we use Choice thresholds.
        # <=40 pages → bs=1, <=80 → bs=2, <=120 → bs=3, etc.
        set_bs = lambda n: sfn.Pass(
            self, f"SetBatchSize{n}",
            result=sfn.Result.from_number(n),
            result_path="$.batch_size",
        )

        prepare_pages = sfn.Pass(
            self, "PreparePages",
            parameters={
                "job_id.$": "$.job_id",
                "user_id.$": "$.user_id",
                "total_pages.$": "$.total_pages",
                "batch_size.$": "$.batch_size",
                "page_indices.$": "States.ArrayRange(0, States.MathAdd($.total_pages, -1), $.batch_size)",
            },
        )

        choose_batch_size = sfn.Choice(self, "ChooseBatchSize")
        choose_batch_size.when(
            sfn.Condition.number_less_than_equals("$.total_pages", 40),
            set_bs(1).next(prepare_pages),
        ).when(
            sfn.Condition.number_less_than_equals("$.total_pages", 80),
            set_bs(2).next(prepare_pages),
        ).when(
            sfn.Condition.number_less_than_equals("$.total_pages", 120),
            set_bs(3).next(prepare_pages),
        ).when(
            sfn.Condition.number_less_than_equals("$.total_pages", 160),
            set_bs(4).next(prepare_pages),
        ).when(
            sfn.Condition.number_less_than_equals("$.total_pages", 200),
            set_bs(5).next(prepare_pages),
        ).otherwise(
            set_bs(6).next(prepare_pages),
        )

        process_batch_task = tasks.LambdaInvoke(
            self, "ProcessBatchTask",
            lambda_function=lambda_version,
            output_path="$.Payload",
            retry_on_service_exceptions=True,
        ).add_retry(
            errors=["Lambda.TooManyRequestsException", "States.TaskFailed"],
            interval=Duration.seconds(1),
            max_attempts=7,
            backoff_rate=2.0,
        )

        process_batches = sfn.Map(
            self, "ProcessBatches",
            items_path="$.page_indices",
            max_concurrency=40,
            parameters={
                "job_id.$": "$.job_id",
                "user_id.$": "$.user_id",
                "start_page.$": "$$.Map.Item.Value",
                "total_pages.$": "$.total_pages",
                "batch_size.$": "$.batch_size",
            },
            result_path="$.parts",
        )

        process_batches.iterator(process_batch_task)

        send_to_gpu = tasks.SqsSendMessage(
            self, "SendToGPU",
            queue=gpu_queue,
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            message_body=sfn.TaskInput.from_object({
                "task_token": sfn.JsonPath.task_token,
                "job_id.$": "$.job_id",
                "user_id.$": "$.user_id",
                "total_pages.$": "$.total_pages",
                "parts.$": "$.parts",
            }),
            heartbeat=Duration.seconds(3600),
        )

        success = sfn.Succeed(self, "ProcessingComplete")

        prepare_pages.next(process_batches).next(send_to_gpu).next(success)
        definition = choose_batch_size

        state_machine = sfn.StateMachine(
            self, "PdfProcessingStateMachine",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.hours(2),
        )

        pdf_processor.grant_invoke(state_machine.role)

        # VPC for ECS GPU cluster (fully private — no NAT gateway, no internet)
        vpc = ec2.Vpc(
            self, "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                ),
            ],
        )

        # VPC endpoints — AWS services over private network, no internet needed
        # Gateway endpoint (free)
        vpc.add_gateway_endpoint("S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )
        # Interface endpoints (~$7/mo each per AZ)
        for svc_name, svc in [
            ("SQS", ec2.InterfaceVpcEndpointAwsService.SQS),
            ("StepFunctions", ec2.InterfaceVpcEndpointAwsService.STEP_FUNCTIONS),
            ("EcrApi", ec2.InterfaceVpcEndpointAwsService.ECR),
            ("EcrDocker", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER),
            ("CloudWatchLogs", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
            ("EcsAgent", ec2.InterfaceVpcEndpointAwsService.ECS_AGENT),
            ("EcsTelemetry", ec2.InterfaceVpcEndpointAwsService.ECS_TELEMETRY),
            ("Ecs", ec2.InterfaceVpcEndpointAwsService.ECS),
        ]:
            vpc.add_interface_endpoint(svc_name,
                service=svc,
                private_dns_enabled=True,
            )

        # ECS cluster
        cluster = ecs.Cluster(self, "GpuCluster", vpc=vpc)

        # Auto Scaling Group with GPU instances
        gpu_asg = autoscaling.AutoScalingGroup(
            self, "GpuAsg",
            vpc=vpc,
            instance_type=ec2.InstanceType("g5.2xlarge"),
            machine_image=ecs.EcsOptimizedImage.amazon_linux2023(
                hardware_type=ecs.AmiHardwareType.GPU,
            ),
            min_capacity=1,
            max_capacity=1,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED),
            update_policy=autoscaling.UpdatePolicy.replacing_update(),
        )

        capacity_provider = ecs.AsgCapacityProvider(
            self, "GpuCapacityProvider",
            auto_scaling_group=gpu_asg,
            enable_managed_scaling=True,
            enable_managed_termination_protection=False,
        )
        cluster.add_asg_capacity_provider(capacity_provider)

        # ECS Task Definition (EC2 launch type for GPU)
        task_def = ecs.Ec2TaskDefinition(
            self, "GpuTaskDef",
            network_mode=ecs.NetworkMode.AWS_VPC,
        )

        task_def.add_container(
            "gpu-processor",
            image=ecs.ContainerImage.from_asset(
                directory=".",
                file="processor/Dockerfile",
            ),
            gpu_count=1,
            memory_limit_mib=28672,
            cpu=4096,
            environment={
                "DOCUMENTS_BUCKET": documents_bucket.bucket_name,
                "SQS_QUEUE_URL": gpu_queue.queue_url,
                "DEVICE": "cuda",
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="gpu-processor",
                log_retention=logs.RetentionDays.ONE_WEEK,
            ),
        )

        # IAM grants for GPU task
        documents_bucket.grant_read_write(task_def.task_role)
        gpu_queue.grant_consume_messages(task_def.task_role)
        task_def.task_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "states:SendTaskSuccess",
                "states:SendTaskFailure",
                "states:SendTaskHeartbeat",
            ],
            resources=["*"],
        ))

        # ECS Service (long-running, polls SQS)
        ecs.Ec2Service(
            self, "GpuService",
            cluster=cluster,
            task_definition=task_def,
            desired_count=1,
            min_healthy_percent=0,
            circuit_breaker=ecs.DeploymentCircuitBreaker(
                enable=True,
                rollback=True,
            ),
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(
                    capacity_provider=capacity_provider.capacity_provider_name,
                    weight=1,
                ),
            ],
        )

        # Outputs
        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
        CfnOutput(self, "DocumentsBucketName", value=documents_bucket.bucket_name)
        CfnOutput(self, "GpuQueueUrl", value=gpu_queue.queue_url)
        CfnOutput(self, "DLQUrl", value=gpu_dlq.queue_url)
