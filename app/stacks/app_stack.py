from aws_cdk import Stack
from constructs import Construct
from stacks.ecr_construct import EcrRepo


class AppStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ecr_repo = EcrRepo(self, "ecr_repo",
                           ecr_repo_name="test_tenzin",
                           ecr_tag_name="v0.0.1",
                           ecr_image_name="test_tenzin",
                           source_code_path='../imap_processing')
        ecr_repo.create_ecr_repo()
        image = ecr_repo.build_and_push_latest_image()
        print(image)
