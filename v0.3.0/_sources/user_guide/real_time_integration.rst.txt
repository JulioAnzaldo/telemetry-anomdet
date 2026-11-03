Real-Time Integration
=====================

The toolkit can run as a Kafka microservice for live telemetry streams.

Workflow:
    CCSDS frames decoded by COSMOS/Yamcs/CCSDSPy
    Decoded telemetry published to Kafka topics
    Anomaly detector subscribes, preprocesses, and publishes scores
    Results viewed in Open MCT, COSMOS, or Grafana dashboards