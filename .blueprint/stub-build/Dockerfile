# Stub image built solely to satisfy BPF's build-container dependency.
# BPF requires any deploy-compose job to be preceded by a build-container
# job. This blueprint consumes only pre-built NGC images, so we build a
# ~2KB busybox-based stub that is not consumed by the compose deployment.
FROM busybox:1.36
LABEL org.opencontainers.image.source="pdnguyen/ai-virtual-assistant"
LABEL org.opencontainers.image.description="BPF dependency sentinel -- not used at runtime"
CMD ["/bin/true"]
