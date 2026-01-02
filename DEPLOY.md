# AWS Deployment

## 1. Launch EC2
```bash
# Create security group
aws ec2 create-security-group --group-name ryan-gpt --description "Ryan-GPT"
aws ec2 authorize-security-group-ingress --group-name ryan-gpt --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name ryan-gpt --protocol tcp --port 80 --cidr 0.0.0.0/0

# Launch instance (Ubuntu 24.04)
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.small \
  --key-name your-key-name \
  --security-groups ryan-gpt \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20}}]' \
  --query 'Instances[0].InstanceId' --output text

# Get public IP
aws ec2 describe-instances --instance-ids <instance-id> --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

## 2. Deploy
```bash
ssh -i key.pem ubuntu@<ip>
git clone https://github.com/ryanzhou147/ryan-gpt.git && cd ryan-gpt
chmod +x deploy.sh && ./deploy.sh
```