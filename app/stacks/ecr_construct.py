from pathlib import Path

from constructs import Construct

from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sns as sns
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from aws_cdk import Stack
from aws_cdk.aws_ecr_assets import DockerImageAsset, Platform
from cdk_ecr_deployment import ECRDeployment, DockerImageName



class EcrRepo(Construct):
    def __init__(self, scope: Construct, id, ecr_repo_name, ecr_image_name, ecr_tag_name, source_code_path, tag_immutability=True, **kwargs) -> None:
        super().__init__(scope, id)
        """Create ECR repo with given inputs.

        Parameters
        ----------
        ecr_repo_name : str
            ECR repo name
        ecr_tag_name: str
            ECR tag name
        source_code_path: IRole
            Code path from which the image is to be pulled from.
        tag_immutability: bool
            The tag mutability setting for the repository. If this parameter is omitted,
            the default setting of MUTABLE will be used which will allow image tags to be overwritten.
            Default: TagMutability.MUTABLE
            Note: If set to IMMUTABLE, image tags will not be overwritten.
        """
        self.ecr_repo_name = ecr_repo_name
        self.ecr_image_name = ecr_image_name
        self.ecr_tag_name = ecr_tag_name
        self.source_code_path = source_code_path
        self.tag_immutability = tag_immutability

    def create_ecr_repo(self):
        """Create ECR repo

        Returns
        -------
        CDK object
            ECR repo object
        """
        ecr_repo = ecr.Repository(self, self.ecr_repo_name,
                                  repository_name=self.ecr_repo_name,
                                  image_tag_mutability=ecr.TagMutability.IMMUTABLE if self.tag_immutability else ecr.TagMutability.MUTABLE)
        return ecr_repo

    def build_and_push_latest_image(self):
        """Build and push latest image

        Returns
        -------
        CDK object
            ECR image object
        """
        ecr_image = DockerImageAsset(self, f'{self.ecr_repo_name}Asset',
                                     directory=self.source_code_path,
                                     file='Dockerfile',
                                     build_args={
                                         'ECR_REPO_NAME': self.ecr_repo_name,
                                         '-t': self.ecr_tag_name
                                     })
        account_id = '449431850278'
        region = 'us-west-1'
        ECRDeployment(self, "DeployDockerImage1",
                      src=DockerImageName(ecr_image.image_uri),
                      dest=DockerImageName(f"{account_id}.dkr.ecr.{region}.amazonaws.com/{self.ecr_repo_name}:{self.ecr_tag_name}")
        )
        return ecr_image
