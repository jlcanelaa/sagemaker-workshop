import boto3
import logging

logging.basicConfig(level=logging.INFO)

s3 = boto3.client('s3')


def lambda_handler(event, context):
    try:
        # check if bucket is not empty
        bucket = event['data']['bucket']
        objects_exist_in_bucket = check_objects_exist_in_bucket(bucket)

        assert objects_exist_in_bucket is True, 'no data found in bucket'
        execution_id = event['execution'].split(':')[-1]  # get step function execution id
        model_prefix = event['training']['model_prefix']
        training_job_name = f'{model_prefix}-{execution_id}'
        return training_job_name

    except Exception as e:
        raise e


def check_objects_exist_in_bucket(bucket):
    try:
        response = s3.list_objects_v2(Bucket=bucket)
        if 'Contents' in response.keys():
            return True
        else:
            return False
    except Exception as e:
        raise e
