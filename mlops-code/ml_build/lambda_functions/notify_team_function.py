import boto3
import os
import json
import logging
logging.basicConfig(level=logging.INFO)

SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']
sns = boto3.client('sns', region_name='eu-west-1')


def lambda_handler(event, context):
    message_body, subject = prepare_message_body(event)
    notify_team(SNS_TOPIC_ARN, subject, message_body)
    return True


def notify_team(topic_arn, subject, message_body):
    try:
        sns.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message_body,
            MessageStructure='json'
        )
    except Exception as e:
        raise e


def prepare_message_body(event):
    subject = 'ML process completed'
    message_body = json.dumps({'default': f"""
                                Process completed.
                                ------------------------------
                                Summary of the process:
                                ------------------------------
                                {event}"""})
    return message_body, subject
