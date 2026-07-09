cd /home/carlos/Clone/uv_vae && \
uv run python scripts/export_clustered_vcf.py \
  --parquet-path /home/carlos/Clone/uv_vae/artifacts/all_out/sampled_deduplicated_variants.parquet \
  --cluster-labels-path /home/carlos/Clone/uv_vae/artifacts/all_out/analysis.parquet \
  --join-on locus \
  --reference GRCh38 \
  --exclude-noise \
  --output-path /home/carlos/Clone/GKclust/test_input/clustered_variants.vcf
