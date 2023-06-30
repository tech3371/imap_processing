# OS
ARG VARIANT=bullseye
FROM --platform=linux/amd64 mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# Updates
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y firefox-esr
RUN sudo apt-get update
RUN sudo apt-get install -y libgtk-3-dev

# Download and install NodeJS which helps to install AWS CDK library
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt-get install -y nodejs
RUN npm install -g aws-cdk@2.86.0

# Download and install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN sudo ./aws/install

# Install Python and pip
RUN sudo apt-get install -y python3 python3-pip

# Install Pip libraries
RUN pip install aws-cdk-lib==2.86.0 
RUN pip install "constructs>=10.0.0,<11.0.0"
RUN pip install boto3==1.26.143
RUN pip install pyyaml==6.0
RUN pip install space_packet_parser==4.0.2
RUN pip install cdk-ecr-deployment==2.5.25

# Set environment and working directory
ENV AWS_CONFIG_FILE=/workspaces/cdk-workspace/.aws/config
ENV AWS_SHARED_CREDENTIALS_FILE=/workspaces/cdk-workspace/.aws/credentials
WORKDIR /workspaces/cdk-workspace/
