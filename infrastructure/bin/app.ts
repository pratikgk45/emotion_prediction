#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { EmotionPredictionStack } from '../lib/emotion-prediction-stack';

const app = new cdk.App();

const domainName = process.env.DOMAIN_NAME;
const hostedZoneId = process.env.HOSTED_ZONE_ID;

new EmotionPredictionStack(app, 'EmotionPredictionStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'us-east-1',
  },
  domainName,
  hostedZoneId,
});
