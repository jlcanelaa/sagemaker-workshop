import json
from aws_cdk import (
    aws_sagemaker as sagemaker,
    core
)


class DeploymentStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        # ==============================
        # ===== Add CFN Parameters =====
        # ==============================
        environment_param = core.CfnParameter(
            scope=self,
            id='Environment',
            type='String'
        )

        sagemaker_execution_role_param = core.CfnParameter(
            scope=self,
            id='SageMakerAPIExecutionRoleArn',
            type='String'
        )

        # ==================================================
        # ===== Parsing Deployment Manifest Parameters =====
        # ==================================================
        with open('deployment_manifest.json') as deployment_manifest:
            content = json.load(deployment_manifest)
        container = content['deployment']['container']
        model_binary_url = content['training']['ModelArtifacts']['S3ModelArtifacts']
        model_name = content['training']['TrainingJobName']
        instance_type = content['deployment']['instance_type']
        instance_count = content['deployment']['instance_count']

        # ===========================
        # ===== SageMaker model =====
        # ===========================
        environment = environment_param.value_as_string
        sagemaker_model_name = f'{model_name}-{environment}'

        container = sagemaker.CfnModel.ContainerDefinitionProperty(
            image=container,
            model_data_url=model_binary_url,
            environment={
                'SAGEMAKER_TFS_NGINX_LOGLEVEL': 'info',
            }
        )

        sagemaker_model = sagemaker.CfnModel(
            scope=self,
            id='Model',
            execution_role_arn=sagemaker_execution_role_param.value_as_string,
            containers=[container],
            model_name=sagemaker_model_name
        )

        # =====================================
        # ===== SageMaker EndPoint Config =====
        # =====================================
        product_variant = sagemaker.CfnEndpointConfig.ProductionVariantProperty(
            model_name=sagemaker_model.attr_model_name,
            variant_name='variant-1',
            instance_type=instance_type,
            initial_instance_count=int(instance_count),
            initial_variant_weight=1.0
        )

        sagemaker_endpoint_config = sagemaker.CfnEndpointConfig(
            scope=self,
            id='EndpointConfig',
            production_variants=[product_variant],
            endpoint_config_name=sagemaker_model.attr_model_name,
        )

        # ==============================
        # ===== SageMaker EndPoint =====
        # ==============================
        sagemaker_endpoint = sagemaker.CfnEndpoint(
            scope=self,
            id='Endpoint',
            endpoint_config_name=sagemaker_endpoint_config.attr_endpoint_config_name,
            endpoint_name=sagemaker_model.attr_model_name,
        )
