export interface Project {
  id: string;
  title: string;
  description: string;
  image: string;
  tags: string[];
  color: string;
  detailedDescription: string;
  architecture: string;
  codeSnippets: {
    title: string;
    language: string;
    code: string;
  }[];
  keyFeatures: string[];
  metrics: {
    label: string;
    value: string;
  }[];
}

export const projects: Project[] = [
  {
    id: "ai-workflow",
    title: "Agentic AI Workflow System",
    description: "Multi-agent system using Foundation Models for automated data pipeline management. Reduced manual intervention by 70% through intelligent self-healing capabilities.",
    image: "/project-ai-workflow.jpg",
    tags: ["Python", "LLMs", "Agent Architecture", "MLOps"],
    color: "#a78bfa",
    detailedDescription: `An intelligent multi-agent system that autonomously manages data pipelines using Large Language Models. The system monitors pipeline health, diagnoses issues, and implements fixes without human intervention.

The architecture consists of specialized agents working together: a Monitor Agent continuously tracks pipeline metrics, a Diagnostic Agent analyzes failures using LLMs to understand error logs, and a Remediation Agent implements fixes by generating and executing code.

This system reduced manual intervention by 70% and improved pipeline uptime from 94% to 99.2% at Clari, processing 2TB+ of data daily.`,
    architecture: `System Architecture Flow:

Data Pipeline → Monitor Agent → Issue Detection
    ↓
Diagnostic Agent (with LLM Analysis)
    ↓
Remediation Agent → Code Generation → Execute Fix
    ↓
Verify Success → Log & Learn → Back to Monitoring

Key Components:
- Monitor Agent: Continuous health tracking
- Diagnostic Agent: AI-powered root cause analysis
- Remediation Agent: Automated fix implementation
- Learning System: Improves from past incidents`,
    codeSnippets: [
      {
        title: "Monitor Agent Implementation",
        language: "python",
        code: `class MonitorAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.metrics_store = MetricsStore()
        self.alert_threshold = 0.95
    
    async def monitor_pipeline(self, pipeline_id: str):
        """Continuously monitor pipeline health"""
        while True:
            metrics = await self.collect_metrics(pipeline_id)
            
            if self.detect_anomaly(metrics):
                context = self.build_context(metrics)
                diagnosis = await self.llm.analyze(
                    prompt=f"Analyze this pipeline issue: {context}",
                    temperature=0.3
                )
                
                await self.trigger_remediation(diagnosis)
            
            await asyncio.sleep(60)  # Check every minute`
      },
      {
        title: "Remediation Agent with Self-Healing",
        language: "python",
        code: `class RemediationAgent:
    def __init__(self, llm_client, sandbox_env):
        self.llm = llm_client
        self.sandbox = sandbox_env
        self.success_history = []
    
    async def fix_issue(self, diagnosis: Dict) -> bool:
        """Generate and execute fix based on diagnosis"""
        
        # Generate fix code using LLM
        fix_code = await self.llm.generate_fix(
            error=diagnosis['error'],
            context=diagnosis['context'],
            previous_fixes=self.get_relevant_fixes(diagnosis)
        )
        
        # Test in sandbox first
        sandbox_result = await self.sandbox.test(fix_code)
        
        if sandbox_result.success:
            # Apply to production
            prod_result = await self.apply_fix(fix_code)
            
            # Learn from result
            self.success_history.append({
                'diagnosis': diagnosis,
                'fix': fix_code,
                'result': prod_result
            })
            
            return prod_result.success
        
        return False`
      }
    ],
    keyFeatures: [
      "Autonomous error detection and diagnosis using LLM analysis",
      "Self-healing capabilities with code generation and execution",
      "Multi-agent coordination for complex pipeline management",
      "Learning from past incidents to improve future responses",
      "Sandbox testing before production deployment",
      "Real-time monitoring with statistical + ML-based anomaly detection"
    ],
    metrics: [
      { label: "Manual Intervention Reduction", value: "70%" },
      { label: "Pipeline Uptime", value: "99.2%" },
      { label: "Mean Time to Resolution", value: "-65%" },
      { label: "Daily Data Processed", value: "2TB+" }
    ]
  },
  {
    id: "mlops-platform",
    title: "Enterprise MLOps Platform",
    description: "End-to-end ML pipeline from training to deployment. Implemented automated model retraining, drift detection, and canary deployments for production models.",
    image: "/project-mlops.jpg",
    tags: ["MLflow", "Kubernetes", "FastAPI", "Prometheus"],
    color: "#f472b6",
    detailedDescription: `A comprehensive MLOps platform handling the complete machine learning lifecycle from experimentation to production deployment at scale.

The platform provides automated model training pipelines, experiment tracking, model versioning, and production deployment with canary releases. It includes drift detection to automatically trigger retraining when model performance degrades.

Built to support 50+ data scientists and serve 100M+ predictions daily with <50ms latency.`,
    architecture: `MLOps Pipeline Flow:

Data Sources → Feature Store → Training Pipeline
                                      ↓
                                MLflow Registry
                                      ↓
                               Model Validation
                                 ↙        ↘
                           Pass ✓        Fail ✗
                              ↓              ↓
                      Staging Environment   Back to Training
                              ↓
                        A/B Testing
                              ↓
                        Production
                              ↓
                        Monitoring
                              ↓
                  Drift Detection → Auto Retrain

Key Components:
- Feature Store: Consistent data for training/serving
- MLflow: Experiment tracking & model registry
- Kubernetes: Scalable deployment infrastructure
- Monitoring: Real-time performance & drift detection`,
    codeSnippets: [
      {
        title: "Automated Training Pipeline",
        language: "python",
        code: `from mlflow import start_run, log_metric, log_param

def train_model_task(**context):
    """Automated model training with MLflow tracking"""
    
    with start_run(run_name=f"training_{context['ds']}"):
        # Load data
        data = load_feature_store(
            start_date=context['ds'],
            features=MODEL_FEATURES
        )
        
        # Log parameters
        for param, value in HYPERPARAMETERS.items():
            log_param(param, value)
        
        # Train model
        model = XGBClassifier(**HYPERPARAMETERS)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_val, y_val)
        for metric, value in metrics.items():
            log_metric(metric, value)
        
        # Register if better than current production
        if metrics['f1_score'] > get_production_baseline():
            mlflow.sklearn.log_model(model, "model")
            register_model(model, metrics)`
      },
      {
        title: "Canary Deployment Strategy",
        language: "python",
        code: `class CanaryDeployment:
    def __init__(self, prometheus_client):
        self.prometheus = prometheus_client
        self.canary_traffic = 0.05  # Start with 5%
    
    async def deploy_model(self, new_model_version: str):
        """Gradual rollout with monitoring"""
        
        # Deploy to canary pods
        await k8s.deploy_canary(new_model_version)
        
        # Gradually increase traffic
        for traffic_pct in [0.05, 0.1, 0.25, 0.5, 1.0]:
            await self.update_traffic_split(traffic_pct)
            
            # Monitor for 1 hour
            await asyncio.sleep(3600)
            
            # Compare metrics
            canary_metrics = self.prometheus.get_metrics(
                service='canary', duration='1h'
            )
            production_metrics = self.prometheus.get_metrics(
                service='production', duration='1h'
            )
            
            # Rollback if canary performs worse
            if not self.is_canary_healthy(canary_metrics, production_metrics):
                await self.rollback()
                return False
        
        return True`
      }
    ],
    keyFeatures: [
      "Automated ML pipeline with Airflow orchestration",
      "Experiment tracking and model versioning with MLflow",
      "Real-time drift detection across multiple statistical methods",
      "Canary deployments with automatic rollback",
      "Feature store for consistent training and serving",
      "Production monitoring with Prometheus and Grafana"
    ],
    metrics: [
      { label: "Daily Predictions", value: "100M+" },
      { label: "Model Latency", value: "<50ms" },
      { label: "Deployment Time", value: "15min" },
      { label: "Data Scientists Supported", value: "50+" }
    ]
  },
  {
    id: "predictive-analytics",
    title: "Real-time Predictive Analytics",
    description: "Spark-based streaming analytics processing 2TB+ daily data. Built scalable ETL pipelines with automated data quality monitoring.",
    image: "/project-predictive.jpg",
    tags: ["Apache Spark", "Airflow", "AWS", "PostgreSQL"],
    color: "#d1e29d",
    detailedDescription: `High-performance real-time analytics platform built on Apache Spark Structured Streaming, processing terabytes of data daily with sub-minute latency.

The system ingests data from multiple sources (Kafka, S3, databases), performs complex transformations and aggregations, and outputs to data warehouses and real-time dashboards.

Includes comprehensive data quality monitoring, automatic schema evolution handling, and fault-tolerant processing with exactly-once semantics.`,
    architecture: `Streaming Data Flow:

Kafka Streams  ┐
S3 Data Lake   ├─→ Spark Streaming Engine
Databases      ┘           ↓
                    Transform & Aggregate
                            ↓
                    Data Quality Checks
                       ↙          ↘
                  Pass ✓         Fail ✗
                    ↓              ↓
              Output Sinks    Dead Letter Queue
                ↙     ↘             ↓
          Redshift  Dashboard   Alert System
          
Key Features:
- Multi-source ingestion (Kafka, S3, DB)
- Real-time transformations with Spark
- Automated quality validation
- Exactly-once processing semantics
- Fault-tolerant checkpointing`,
    codeSnippets: [
      {
        title: "Spark Structured Streaming Pipeline",
        language: "scala",
        code: `val rawEvents = spark.readStream
  .format("kafka")
  .option("kafka.bootstrap.servers", "kafka:9092")
  .option("subscribe", "user_events")
  .load()

val processedEvents = rawEvents
  .select(from_json(col("value").cast("string"), eventSchema).as("data"))
  .select("data.*")
  .withWatermark("event_time", "10 minutes")
  .groupBy(
    window(col("event_time"), "5 minutes"),
    col("user_id"),
    col("event_type")
  )
  .agg(
    count("*").as("event_count"),
    sum("value").as("total_value"),
    avg("latency").as("avg_latency")
  )

val query = processedEvents.writeStream
  .foreachBatch { (batchDF, batchId) =>
    if (validateBatch(batchDF)) {
      batchDF.write.format("jdbc").save()
    }
  }
  .start()`
      }
    ],
    keyFeatures: [
      "Real-time data processing with Spark Structured Streaming",
      "Exactly-once processing semantics with checkpointing",
      "Automated data quality validation",
      "Multi-sink output (Redshift, S3, dashboards)",
      "Schema evolution handling",
      "Dynamic resource optimization based on workload"
    ],
    metrics: [
      { label: "Daily Data Processed", value: "2TB+" },
      { label: "Processing Latency", value: "<60s" },
      { label: "Data Quality Pass Rate", value: "99.8%" },
      { label: "Uptime", value: "99.95%" }
    ]
  },
  {
    id: "computer-vision",
    title: "Computer Vision Pipeline",
    description: "Automated image processing pipeline with 94% classification accuracy. Implemented ensemble models for industrial fault detection.",
    image: "/project-vision.jpg",
    tags: ["TensorFlow", "OpenCV", "Docker", "REST API"],
    color: "#60a5fa",
    detailedDescription: `Production-ready computer vision system for automated industrial fault detection using ensemble deep learning models.

The pipeline processes images from manufacturing lines in real-time, detecting defects with 94% accuracy. It combines multiple CNN architectures (ResNet, EfficientNet, Vision Transformer) in an ensemble for robust predictions.

Deployed as a containerized microservice handling 1000+ images per minute with automated retraining on misclassified samples.`,
    architecture: `Computer Vision Pipeline:

Image Input → Preprocessing → Augmentation
                                  ↓
                          Ensemble Models
                        ↙       ↓        ↘
                  ResNet50  EfficientNet  ViT
                        ↘       ↓        ↙
                        Voting Classifier
                                ↓
                          Confidence Score
                            ↙        ↘
                    >95% Accept    <95% Human Review
                                          ↓
                                   Retraining Queue
                                          ↓
                                   Improve Model

Components:
- Multi-model ensemble for robustness
- Confidence-based routing
- Active learning pipeline
- FastAPI inference server
- Docker + Kubernetes deployment`,
    codeSnippets: [
      {
        title: "Ensemble Model Architecture",
        language: "python",
        code: `class EnsembleVisionModel:
    def __init__(self):
        self.models = {
            'resnet': self.build_resnet(),
            'efficientnet': self.build_efficientnet(),
            'vit': self.build_vision_transformer()
        }
        self.ensemble = self.build_ensemble()
    
    def build_ensemble(self):
        """Ensemble with weighted voting"""
        input_layer = layers.Input(shape=(224, 224, 3))
        
        # Get predictions from all models
        resnet_pred = self.models['resnet'](input_layer)
        eff_pred = self.models['efficientnet'](input_layer)
        vit_pred = self.models['vit'](input_layer)
        
        # Weighted average
        ensemble_pred = layers.Average()([
            resnet_pred,
            eff_pred,
            vit_pred
        ])
        
        return Model(inputs=input_layer, outputs=ensemble_pred)`
      },
      {
        title: "Real-time Inference API",
        language: "python",
        code: `from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="Defect Detection API")

@app.post("/detect")
async def detect_defect(file: UploadFile = File(...)):
    """Detect defects in uploaded image"""
    
    # Read and preprocess image
    contents = await file.read()
    image = preprocess_image(contents)
    
    # Run inference
    predictions = model.predict(image)
    confidence = float(predictions.max())
    class_name = CLASS_NAMES[predictions.argmax()]
    
    result = {
        'class': class_name,
        'confidence': confidence,
        'requires_review': confidence < 0.95
    }
    
    # Queue for human review if low confidence
    if result['requires_review']:
        await queue_for_review(image, result)
    
    return result`
      }
    ],
    keyFeatures: [
      "Ensemble of multiple CNN architectures for robust predictions",
      "Real-time inference API with FastAPI (1000+ images/min)",
      "Active learning pipeline with human-in-the-loop",
      "Automated retraining on misclassified samples",
      "Confidence-based routing to human review",
      "Containerized deployment with Docker + Kubernetes"
    ],
    metrics: [
      { label: "Classification Accuracy", value: "94%" },
      { label: "Processing Speed", value: "1000+ img/min" },
      { label: "False Positive Rate", value: "<3%" },
      { label: "Model Uptime", value: "99.9%" }
    ]
  }
];
