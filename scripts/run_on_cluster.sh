#!/bin/bash
# Helper script to run commands on the cluster
#
# Usage: ./run_on_cluster.sh <command>

set -e

POD_NAME="mixtral-dev"
NAMESPACE="user-sbowerma"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if pod exists
if ! kubectl get pod "$POD_NAME" -n "$NAMESPACE" &> /dev/null; then
    echo "ERROR: Pod $POD_NAME not found in namespace $NAMESPACE"
    echo "Please create the pod first with: kubectl apply -f deployment/dev-pod.yaml"
    exit 1
fi

# Execute command
if [ $# -eq 0 ]; then
    # No args = interactive shell
    echo "Opening interactive shell in $POD_NAME..."
    kubectl exec -it "$POD_NAME" -n "$NAMESPACE" -- /bin/bash
else
    # Execute provided command
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- "$@"
fi
