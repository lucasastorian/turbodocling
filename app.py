#!/usr/bin/env python3
import os
import aws_cdk as cdk
from turbodocling.turbodocling_stack import TurboStack

app = cdk.App()
stage = app.node.try_get_context("stage") or "dev"

TurboStack(
    app, f"turbodocling-turbo-{stage}",
    stage=stage,
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION"),
    ),
)

app.synth()
