terraform {
  backend "s3" {
    bucket = "tfstate-slurm-k8s-03d60b31e979ded3707457944b273fb3"
    key    = "slurm-k8s.tfstate"

    endpoints = {
      s3 = "https://storage.eu-north2.nebius.cloud:443"
    }
    region = "eu-north2"

    skip_region_validation      = true
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }
}
