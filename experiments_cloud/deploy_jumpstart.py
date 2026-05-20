#!/usr/bin/env python3
"""
Deploy or delete a SageMaker JumpStart endpoint.

Usage:
    # Deploy Qwen 3.5 9B (takes ~10 min, prints endpoint name on success):
    python experiments_cloud/deploy_jumpstart.py --model qwen3-5-9b --profile trgpt

    # Delete when done:
    python experiments_cloud/deploy_jumpstart.py --delete <endpoint-name> --profile trgpt

    # List active JumpStart endpoints:
    python experiments_cloud/deploy_jumpstart.py --list --profile trgpt
"""
import argparse
import sys
import time

# JumpStart model IDs — verify in SageMaker console if deploy fails
MODELS = {
    "qwen3-5-9b": {
        "model_id": "huggingface-llm-qwen3-5-9b-instruct",
        "instance":  "ml.g5.2xlarge",
    },
    "qwen3-5-27b": {
        "model_id": "huggingface-llm-qwen3-5-27b-instruct",
        "instance":  "ml.g5.12xlarge",
    },
    "qwen2-5-7b": {
        "model_id": "huggingface-llm-qwen2-5-7b-instruct",
        "instance":  "ml.g5.2xlarge",
    },
}


def get_sagemaker_session(profile: str, region: str):
    import boto3
    import sagemaker
    session = boto3.Session(profile_name=profile, region_name=region)
    return sagemaker.Session(boto_session=session)


def deploy(model_key: str, profile: str, region: str):
    try:
        import sagemaker
        from sagemaker.jumpstart.model import JumpStartModel
    except ImportError:
        print("ERROR: sagemaker SDK not installed. Run:")
        print("  uv pip install sagemaker --default-index https://pypi.org/simple")
        sys.exit(1)

    info = MODELS.get(model_key)
    if not info:
        print(f"Unknown model key '{model_key}'. Available: {list(MODELS)}")
        sys.exit(1)

    sm_session = get_sagemaker_session(profile, region)
    model = JumpStartModel(
        model_id=info["model_id"],
        sagemaker_session=sm_session,
    )

    print(f"Deploying {info['model_id']} on {info['instance']} ...")
    print("This takes ~10 minutes. Do NOT interrupt.\n")
    t0 = time.time()

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=info["instance"],
    )

    elapsed = int(time.time() - t0)
    print(f"\n✓ Deployed in {elapsed}s")
    print(f"Endpoint name: {predictor.endpoint_name}")
    print(f"\nRun experiments with:")
    print(f"  JUMPSTART_ENDPOINT_NAME={predictor.endpoint_name} \\")
    print(f"  python experiments_cloud/ucurve_sweep.py --model jumpstart-{model_key}")
    print(f"\nDelete when done:")
    print(f"  python experiments_cloud/deploy_jumpstart.py --delete {predictor.endpoint_name} --profile {profile}")


def delete_endpoint(endpoint_name: str, profile: str, region: str):
    import boto3
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("sagemaker")
    client.delete_endpoint(EndpointName=endpoint_name)
    print(f"✓ Deleted endpoint: {endpoint_name}")


def list_endpoints(profile: str, region: str):
    import boto3
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("sagemaker")
    resp = client.list_endpoints(MaxResults=50)
    endpoints = [e for e in resp["Endpoints"] if "jumpstart" in e["EndpointName"].lower()
                 or "huggingface" in e["EndpointName"].lower()]
    if not endpoints:
        print("No JumpStart endpoints found.")
        return
    for e in endpoints:
        print(f"  {e['EndpointStatus']:12s}  {e['EndpointName']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   help="Model key to deploy (e.g. qwen3-5-9b)")
    parser.add_argument("--delete",  metavar="ENDPOINT_NAME", help="Delete this endpoint")
    parser.add_argument("--list",    action="store_true", help="List active JumpStart endpoints")
    parser.add_argument("--profile", default="trgpt", help="AWS profile (default: trgpt)")
    parser.add_argument("--region",  default="us-east-1")
    args = parser.parse_args()

    if args.delete:
        delete_endpoint(args.delete, args.profile, args.region)
    elif args.list:
        list_endpoints(args.profile, args.region)
    elif args.model:
        deploy(args.model, args.profile, args.region)
    else:
        parser.print_help()
