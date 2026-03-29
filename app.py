#!/usr/bin/env python3
import os
import aws_cdk as cdk
from turbodocling.turbodocling_stack import TurboStack

app = cdk.App()
stage = app.node.try_get_context("stage") or "dev"
pdfium_variant = (app.node.try_get_context("pdfium_variant") or "upstream").lower()
pypdfium2_wheel_url = app.node.try_get_context("pypdfium2_wheel_url") or ""

TurboStack(
    app, f"turbodocling-turbo-{stage}",
    stage=stage,
    pdfium_variant=pdfium_variant,
    pypdfium2_wheel_url=pypdfium2_wheel_url,
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION"),
    ),
)

app.synth()
