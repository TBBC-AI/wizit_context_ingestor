import logging
import os
from typing import Optional

from boto3 import client as boto3_client
from botocore.exceptions import ClientError

from ...application.interfaces import PersistenceService
from ...domain.models import ParsedDoc

logger = logging.getLogger(__name__)


class S3StorageService(PersistenceService):
    """Persistence service for S3 storage."""

    __slots__ = ("origin_bucket_name", "target_bucket_name", "region_name")

    def __init__(
        self,
        origin_bucket_name: str,
        target_bucket_name: str,
        region_name: str = "us-east-1",
    ):
        self.s3 = boto3_client("s3", region_name=region_name)
        self.origin_bucket_name = origin_bucket_name
        self.target_bucket_name = target_bucket_name
        self.supports_tagging = hasattr(self, "retrieve_file_tags")

    def load_markdown_file_content(self, file_key: str) -> tuple[str, dict]:
        """Load markdown file content from S3 storage.

        Args:
            file_key: The key (path) of the file in S3

        Returns:
            tuple: A tuple containing the file content as a string and the metadata dictionary

        Raises:
            ClientError: If there's an error retrieving the object from S3
        """
        try:
            # Get the object from S3
            response = self.s3.get_object(Bucket=self.target_bucket_name, Key=file_key)
            file_content = response["Body"].read()
            # tmp_file_key = f"/tmp/{file_key}"
            # os.makedirs(os.path.dirname(tmp_file_key), exist_ok=True)
            # with open(tmp_file_key, "wb") as f:
            #     f.write(response["Body"].read())
            # with open(tmp_file_key, "r", encoding="utf-8") as f:
            #     file_content = f.read()
            metadata = response["Metadata"] if response.get("Metadata") else {}
            return file_content, metadata
        except ClientError as e:
            logger.error(f"Error loading file {file_key} from S3: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading file {file_key} from S3: {str(e)}")
            raise

    def retrieve_raw_file(self, file_key: str) -> tuple:
        """Retrieve file path in tmp folder from S3 storage.

        Args:
            file_key: The key (path) of the file in S3

        Returns:
            tuple: The path of the file in tmp folder and the file metadata

        Raises:
            ClientError: If there's an error retrieving the object from S3
        """
        try:
            # Get the object from S3
            response = self.s3.get_object(Bucket=self.origin_bucket_name, Key=file_key)
            tmp_file_key = f"/tmp/{file_key}"
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(tmp_file_key), exist_ok=True)
            with open(tmp_file_key, "wb") as f:
                f.write(response["Body"].read())
            metadata = response["Metadata"] if response.get("Metadata") else {}
            return tmp_file_key, metadata

        except ClientError as e:
            logger.error(f"Error retrieving file {file_key} from S3: {str(e)}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving file {file_key} from S3: {str(e)}"
            )
            raise

    def save_parsed_document(
        self,
        file_key: str,
        parsed_document: ParsedDoc,
        file_tags: Optional[dict] = {},
        metadata: Optional[dict] = {},
    ):
        """Save a parsed document to S3.

        Args:
            file_name: The key (path) to save the file to in S3
            parsed_document: The parsed document to save
            file_tags: Tags to add to parsed document
            metadata: Metadata to add to parsed document

        Raises:
            ClientError: If there's an error saving to S3
        """
        try:
            # Convert document content to bytes
            content_bytes = parsed_document.document_text.encode("utf-8")
            # Upload the file to S3
            put_kwargs = {
                "Bucket": self.target_bucket_name,
                "Key": file_key,
                "Body": content_bytes,
                "Metadata": metadata,
            }
            if file_tags:
                tagging_string = "&".join(
                    [f"{key}={value}" for key, value in file_tags.items()]
                )
                put_kwargs["Tagging"] = tagging_string

            self.s3.put_object(**put_kwargs)
            logger.info(f"Successfully saved document to S3 as {file_key}")
        except ClientError as e:
            logger.error(f"Error saving document to S3: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving document to S3: {str(e)}")
            raise

    def retrieve_file_tags(self, file_key: str, bucket_name: str):
        """Retrieve a file tagging dict

        Args:
            file_key: The key (path) to retrieve tags
        """
        try:
            response = self.s3.get_object_tagging(Bucket=bucket_name, Key=file_key)
            if response["TagSet"] and len(response["TagSet"]) > 0:
                logger.info("Successfully retrieved file tags from S3")
                return {item["Key"]: item["Value"] for item in response["TagSet"]}
            else:
                logger.info(f"No tags found for file {file_key}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving file tags from S3: {str(e)}")
            return None
