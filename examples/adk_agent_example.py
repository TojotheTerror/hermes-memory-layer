"""Full ADK agent example — deployable to Agent Platform Runtime."""
import os

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0810135629")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# This file is reference / deploy script — it won't run without google-adk + GCP creds.
# To deploy:
#   pip install -e ".[adk]"
#   export GOOGLE_CLOUD_PROJECT=...
#   export GOOGLE_CLOUD_LOCATION=us-central1
#   export STAGING_BUCKET=gs://your-bucket
#   python examples/adk_agent_example.py

try:
    from hermes_memory.adk_integration import build_memory_agent, build_adk_app
    from hermes_memory.config import load_config
    import vertexai

    cfg = load_config(project=PROJECT, location=LOCATION)
    print(f"Building ADK agent: project={cfg.project} location={cfg.location}")

    agent, runner = build_memory_agent(project=cfg.project, location=cfg.location, agent_engine_id=cfg.agent_engine_id)
    print(f"Agent: {agent.name} model={agent.model} tools={[t.__class__.__name__ for t in agent.tools]}")

    # Optional: deploy to Agent Platform Runtime
    STAGING_BUCKET = os.environ.get("STAGING_BUCKET")
    if STAGING_BUCKET and cfg.agent_engine_id:
        client = vertexai.Client(project=cfg.project, location=cfg.location)
        adk_app = build_adk_app(agent=agent, project=cfg.project, location=cfg.location, agent_engine_id=cfg.agent_engine_id)
        print(f"Deploying to Agent Platform Runtime (staging: {STAGING_BUCKET})...")
        engine = client.agent_engines.create(
            agent_engine=adk_app,
            config={"staging_bucket": STAGING_BUCKET, "requirements": ["google-cloud-aiplatform[agent_engines,adk]"]},
        )
        print(f"Deployed: {engine.api_resource.name}")
    else:
        print("Skipping deploy — set STAGING_BUCKET and GOOGLE_CLOUD_AGENT_ENGINE_ID to deploy.")
        print("Local test: use runner to chat with the agent in-process.")

except ImportError as e:
    print(f"ADK not installed: {e}")
    print("pip install 'hermes-memory-layer[adk]'")
except Exception as e:
    print(f"Error: {e}")
    import traceback; traceback.print_exc()
