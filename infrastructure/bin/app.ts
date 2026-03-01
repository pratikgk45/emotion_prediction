#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { EmotionPredictionStack } from '../lib/emotion-prediction-stack';

const app = new cdk.App();

new EmotionPredictionStack(app, 'EmotionPredictionStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'us-east-1',
  },
});
