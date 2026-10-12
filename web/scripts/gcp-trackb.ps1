# GCP Track B test-and-drop. Run from Windows PowerShell with gcloud installed.
# Free-tier compliant: us-central1, e2-micro, 30GB STANDARD disk, ephemeral IP.
param(
  [string]$Project = $env:GCP_PROJECT,
  [string]$Zone = "us-central1-a",
  [string]$Vm = "agricnxedge-trackb",
  [string]$Action = "create"
)

if (-not $Project) { throw "Set -Project or env GCP_PROJECT first: gcloud config get-value project" }

if ($Action -eq "create") {
  gcloud config set project $Project
  gcloud compute instances create $Vm `
    --zone=$Zone `
    --machine-type=e2-micro `
    --provisioning-model=STANDARD `
    --image-family=ubuntu-2204-lts `
    --image-project=ubuntu-os-cloud `
    --boot-disk-size=30GB `
    --boot-disk-type=pd-standard `
    --tags=http-server,https-server `
    --metadata=enable-oslogin=false
  gcloud compute firewall-rules describe allow-http --format="value(name)" 2>$null
  if (-not $?) {
    gcloud compute firewall-rules create allow-http --allow=tcp:80 --target-tags=http-server --description="Track B demo"
  }
  gcloud compute instances describe $Vm --zone=$Zone --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
  Write-Output "Copy the IP above into ansible/inventory.gcp.ini, then run Ansible from WSL."
}

if ($Action -eq "delete") {
  gcloud compute instances delete $Vm --zone=$Zone --quiet
  Write-Output "Verify billing stopped: gcloud compute instances list; gcloud compute disks list; gcloud compute addresses list"
}
