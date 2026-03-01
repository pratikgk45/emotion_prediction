import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as ecr_assets from 'aws-cdk-lib/aws-ecr-assets';
import { Construct } from 'constructs';
import * as path from 'path';

export interface EmotionPredictionStackProps extends cdk.StackProps {
  domainName?: string;
  hostedZoneId?: string;
}

export class EmotionPredictionStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: EmotionPredictionStackProps) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, 'EmotionVpc', {
      maxAzs: 2,
      natGateways: 1,
    });

    const cluster = new ecs.Cluster(this, 'EmotionCluster', {
      vpc,
      containerInsights: true,
    });

    const image = new ecr_assets.DockerImageAsset(this, 'EmotionImage', {
      directory: path.join(__dirname, '../../'),
    });

    const taskDefinition = new ecs.FargateTaskDefinition(this, 'EmotionTask', {
      memoryLimitMiB: 1024,
      cpu: 512,
    });

    const container = taskDefinition.addContainer('EmotionContainer', {
      image: ecs.ContainerImage.fromDockerImageAsset(image),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: 'emotion' }),
    });

    container.addPortMappings({ containerPort: 8080 });

    const service = new ecs.FargateService(this, 'EmotionService', {
      cluster,
      taskDefinition,
      desiredCount: 1,
    });

    const alb = new elbv2.ApplicationLoadBalancer(this, 'EmotionALB', {
      vpc,
      internetFacing: true,
    });

    const httpListener = alb.addListener('HttpListener', { port: 80 });

    const targetGroup = new elbv2.ApplicationTargetGroup(this, 'EmotionTarget', {
      vpc,
      port: 8080,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targets: [service],
      healthCheck: {
        path: '/',
        interval: cdk.Duration.seconds(60),
        timeout: cdk.Duration.seconds(30),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 3,
      },
    });

    // If domain and hosted zone provided, add HTTPS
    if (props?.domainName && props?.hostedZoneId) {
      const hostedZone = route53.HostedZone.fromHostedZoneAttributes(this, 'HostedZone', {
        hostedZoneId: props.hostedZoneId,
        zoneName: props.domainName,
      });

      const certificate = new acm.Certificate(this, 'Certificate', {
        domainName: props.domainName,
        validation: acm.CertificateValidation.fromDns(hostedZone),
      });

      // HTTPS listener
      const httpsListener = alb.addListener('HttpsListener', {
        port: 443,
        certificates: [certificate],
      });

      httpsListener.addTargetGroups('HttpsTargetGroup', {
        targetGroups: [targetGroup],
      });

      // Redirect HTTP to HTTPS
      httpListener.addAction('RedirectToHttps', {
        action: elbv2.ListenerAction.redirect({
          protocol: 'HTTPS',
          port: '443',
          permanent: true,
        }),
      });

      // Create Route53 A record
      new route53.ARecord(this, 'AliasRecord', {
        zone: hostedZone,
        target: route53.RecordTarget.fromAlias({
          bind: () => ({
            dnsName: alb.loadBalancerDnsName,
            hostedZoneId: alb.loadBalancerCanonicalHostedZoneId,
          }),
        }),
      });

      new cdk.CfnOutput(this, 'DomainURL', {
        value: `https://${props.domainName}`,
        description: 'Application URL',
      });
    } else {
      // No HTTPS, just forward HTTP traffic
      httpListener.addTargetGroups('HttpTargetGroup', {
        targetGroups: [targetGroup],
      });

      new cdk.CfnOutput(this, 'LoadBalancerURL', {
        value: `http://${alb.loadBalancerDnsName}`,
        description: 'Application URL (HTTP only - HTTPS requires domain)',
      });
    }

    new cdk.CfnOutput(this, 'LoadBalancerDNS', {
      value: alb.loadBalancerDnsName,
      description: 'Load Balancer DNS',
    });
  }
}
