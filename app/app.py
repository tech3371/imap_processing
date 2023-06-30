#!/usr/bin/env python3
import os

import aws_cdk as cdk

from stacks.app_stack import AppStack


app = cdk.App()

AppStack(app, "AppStack")

app.synth()
