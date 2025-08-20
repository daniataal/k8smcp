"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { Play, Calendar, Clock, CheckCircle, XCircle, Eye, Plus } from "lucide-react"

const activeWorkflows = [
  {
    id: "wf-001",
    name: "MNIST Training Pipeline",
    status: "Running",
    progress: 65,
    startTime: "2024-01-15 14:30",
    estimatedEnd: "2024-01-15 16:45",
  },
  {
    id: "wf-002",
    name: "Sentiment Model Deployment",
    status: "Succeeded",
    progress: 100,
    startTime: "2024-01-15 13:15",
    estimatedEnd: "2024-01-15 14:20",
  },
  {
    id: "wf-003",
    name: "Image Classifier Training",
    status: "Failed",
    progress: 45,
    startTime: "2024-01-15 12:00",
    estimatedEnd: "2024-01-15 15:30",
  },
]

export function WorkflowManagement() {
  const [selectedModelType, setSelectedModelType] = useState("")
  const [selectedFramework, setSelectedFramework] = useState("")
  const [customModelSpec, setCustomModelSpec] = useState("")
  const [customFrameworkSpec, setCustomFrameworkSpec] = useState("")
  const [epochs, setEpochs] = useState("10")
  const [batchSize, setBatchSize] = useState("32")
  const [replicas, setReplicas] = useState("3")
  const [deployOnSuccess, setDeployOnSuccess] = useState(true)

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold mb-2">Workflow Management</h2>
        <p className="text-muted-foreground">Create, monitor, and manage your MLOps workflows</p>
      </div>

      <Tabs defaultValue="create" className="space-y-6">
        <TabsList>
          <TabsTrigger value="create">Create Workflow</TabsTrigger>
          <TabsTrigger value="active">Active Workflows</TabsTrigger>
        </TabsList>

        <TabsContent value="create" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Plus className="w-5 h-5" />
                Create New Workflow
              </CardTitle>
              <CardDescription>Configure and launch a new MLOps pipeline without writing code</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Model Configuration */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Model Configuration</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="framework">Framework</Label>
                    <Select value={selectedFramework} onValueChange={setSelectedFramework}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select ML framework" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pytorch">PyTorch</SelectItem>
                        <SelectItem value="tensorflow">TensorFlow</SelectItem>
                        <SelectItem value="scikit-learn">Scikit-learn</SelectItem>
                        <SelectItem value="xgboost">XGBoost</SelectItem>
                        <SelectItem value="lightgbm">LightGBM</SelectItem>
                        <SelectItem value="huggingface">Hugging Face Transformers</SelectItem>
                        <SelectItem value="keras">Keras</SelectItem>
                        <SelectItem value="custom">Custom Framework</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="model-type">Model Type</Label>
                    <Select value={selectedModelType} onValueChange={setSelectedModelType}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select model type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="mnist_pytorch_cnn">MNIST PyTorch CNN</SelectItem>
                        <SelectItem value="sentiment_transformer">Sentiment Transformer</SelectItem>
                        <SelectItem value="image_resnet">Image ResNet</SelectItem>
                        <SelectItem value="text_classification">Text Classification</SelectItem>
                        <SelectItem value="custom">Custom Model (AI Generated)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {selectedFramework === "custom" && (
                  <div className="space-y-2">
                    <Label htmlFor="custom-framework-spec">Custom Framework Specification</Label>
                    <Textarea
                      id="custom-framework-spec"
                      placeholder="Describe your custom framework requirements. For example: 'Use JAX with Flax for neural network training, with custom optimizers and distributed training support across multiple GPUs.'"
                      value={customFrameworkSpec}
                      onChange={(e) => setCustomFrameworkSpec(e.target.value)}
                      rows={3}
                      className="resize-none"
                    />
                    <p className="text-xs text-muted-foreground">
                      Specify your custom framework setup and requirements.
                    </p>
                  </div>
                )}

                {selectedModelType === "custom" && (
                  <div className="space-y-2">
                    <Label htmlFor="custom-model-spec">Custom Model Specification</Label>
                    <Textarea
                      id="custom-model-spec"
                      placeholder="Describe your model requirements in natural language. For example: 'Create a transformer model for sentiment analysis with BERT architecture, fine-tuned for financial text, with attention visualization and custom loss function for imbalanced classes.'"
                      value={customModelSpec}
                      onChange={(e) => setCustomModelSpec(e.target.value)}
                      rows={4}
                      className="resize-none"
                    />
                    <p className="text-xs text-muted-foreground">
                      Our AI will generate the complete model architecture and training code based on your description.
                    </p>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="namespace">Namespace</Label>
                    <Input id="namespace" placeholder="default" />
                  </div>
                </div>
              </div>

              {/* Training Parameters */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Training Parameters</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="epochs">Epochs</Label>
                    <Input id="epochs" type="number" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="batch-size">Batch Size</Label>
                    <Input
                      id="batch-size"
                      type="number"
                      value={batchSize}
                      onChange={(e) => setBatchSize(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="learning-rate">Learning Rate</Label>
                    <Input id="learning-rate" placeholder="0.001" />
                  </div>
                </div>
              </div>

              {/* Data Configuration */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Data Configuration</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="data-source">Data Source</Label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select data source" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="dvc">DVC</SelectItem>
                        <SelectItem value="s3">Amazon S3</SelectItem>
                        <SelectItem value="gcs">Google Cloud Storage</SelectItem>
                        <SelectItem value="local">Local Path</SelectItem>
                        <SelectItem value="database">Database</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="data-path">Data Path</Label>
                    <Input id="data-path" placeholder="data/mnist" />
                  </div>
                </div>
              </div>

              {/* Deployment Parameters */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Deployment Configuration</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="replicas">Replicas</Label>
                    <Input id="replicas" type="number" value={replicas} onChange={(e) => setReplicas(e.target.value)} />
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="deploy-on-success" checked={deployOnSuccess} onCheckedChange={setDeployOnSuccess} />
                    <Label htmlFor="deploy-on-success">Deploy if training succeeds</Label>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-4 pt-4">
                <Button className="flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  Run Now
                </Button>
                <Button variant="outline" className="flex items-center gap-2 bg-transparent">
                  <Calendar className="w-4 h-4 mr-1" />
                  Schedule Workflow
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="active" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Active Workflows</CardTitle>
              <CardDescription>Monitor the status and progress of your running workflows</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {activeWorkflows.map((workflow) => (
                  <div key={workflow.id} className="p-4 border rounded-lg space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-semibold">{workflow.name}</h4>
                        <p className="text-sm text-muted-foreground">ID: {workflow.id}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            workflow.status === "Running"
                              ? "default"
                              : workflow.status === "Succeeded"
                                ? "secondary"
                                : "destructive"
                          }
                        >
                          {workflow.status === "Running" && <Clock className="w-3 h-3 mr-1" />}
                          {workflow.status === "Succeeded" && <CheckCircle className="w-3 h-3 mr-1" />}
                          {workflow.status === "Failed" && <XCircle className="w-3 h-3 mr-1" />}
                          {workflow.status}
                        </Badge>
                        <Button variant="outline" size="sm">
                          <Eye className="w-4 h-4 mr-1" />
                          View Logs
                        </Button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Started:</span>
                        <span className="ml-2">{workflow.startTime}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Est. End:</span>
                        <span className="ml-2">{workflow.estimatedEnd}</span>
                      </div>
                    </div>

                    {workflow.status === "Running" && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Progress</span>
                          <span>{workflow.progress}%</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div
                            className="bg-primary h-2 rounded-full transition-all duration-300"
                            style={{ width: `${workflow.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
