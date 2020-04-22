import boto3
import logging

logging.basicConfig(level=logging.INFO)

cloudwatch_client = boto3.client('logs')


def lambda_handler(event, context):
    try:
        training_job_name = event['training']['TrainingJobName']
        log_group_name = event['validation']['log_group_name']
        validation_metric = event['validation']['validation_metric']
        validation_minimum_value = float(event['validation']['validation_minimum_value'])

        logging.info('Get log stream name from training job name...')
        log_stream_name = get_log_stream_name(log_group_name, training_job_name)
        logging.info('Evaluating model...')
        metric_value = get_cloudwatch_logs_and_evaluate_model(log_group_name, log_stream_name, validation_metric,
                                                              validation_minimum_value)

        assert metric_value >= validation_minimum_value, f'model not validated. metric_value: {metric_value}'
        return {
            'validated': 'true',
            'metric_value': metric_value,
        }

    except Exception as e:
        raise e


def get_log_stream_name(log_group_name, log_stream_name_prefix):
    try:
        response = cloudwatch_client.describe_log_streams(
            logGroupName=log_group_name,
            logStreamNamePrefix=log_stream_name_prefix
        )
        log_stream_name = response['logStreams'][0]['logStreamName']
        return log_stream_name

    except Exception as e:
        logging.error('Unable to get the log_stream_name.')
        raise e


def get_cloudwatch_logs_and_evaluate_model(log_group_name, log_stream_name, validation_metric,
                                           validation_minimum_value):
    try:
        response = cloudwatch_client.filter_log_events(logGroupName=log_group_name,
                                                       logStreamNames=[log_stream_name],
                                                       filterPattern=validation_metric)
        metric_log = response['events'][0]['message']
        metric_value = float(metric_log.split(': ')[1]) * 100
        return metric_value

    except Exception as e:
        logging.error('Unable to validate the model.')
        raise e
