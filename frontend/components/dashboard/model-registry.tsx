import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Brain, Eye, Rocket, TrendingUp, ArrowUp, RotateCcw } from "lucide-react"

const registeredModels = [
  {
    id: "model-001",
    name: "MNIST CNN Classifier",
    version: "v2.1",
    status: "Production",
    accuracy: 98.5,
    precision: 97.8,
    recall: 98.2,
    registeredAt: "2024-01-10",
    jobId: "job-abc123",
    description: "Convolutional neural network for handwritten digit recognition",
  },
  {
    id: "model-002",
    name: "Sentiment Analysis BERT",
    version: "v1.3",
    status: "Staging",
    accuracy: 94.2,
    precision: 93.8,
    recall: 94.6,
    registeredAt: "2024-01-12",
    jobId: "job-def456",
    description: "BERT-based model for sentiment classification",
  },
  {
    id: "model-003",
    name: "Image Classification ResNet",
    version: "v0.9",
    status: "Development",
    accuracy: 91.8,
    precision: 90.5,
    recall: 92.1,
    registeredAt: "2024-01-14",
    jobId: "job-ghi789",
    description: "ResNet architecture for general image classification",
  },
  {
    id: "model-004",
    name: "Text Classification RoBERTa",
    version: "v1.0",
    status: "Archived",
    accuracy: 89.3,
    precision: 88.7,
    recall: 89.9,
    registeredAt: "2024-01-05",
    jobId: "job-jkl012",
    description: "RoBERTa model for multi-class text classification",
  },
]

export function ModelRegistry() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold mb-2">Model Registry</h2>
        <p className="text-muted-foreground">Centralized repository for all your machine learning models</p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Models</p>
                <p className="text-2xl font-bold">12</p>
              </div>
              <Brain className="w-8 h-8 text-primary" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">In Production</p>
                <p className="text-2xl font-bold">4</p>
              </div>
              <Rocket className="w-8 h-8 text-chart-4" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">In Staging</p>
                <p className="text-2xl font-bold">3</p>
              </div>
              <TrendingUp className="w-8 h-8 text-chart-1" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg Accuracy</p>
                <p className="text-2xl font-bold">93.5%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-chart-2" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model List */}
      <div className="grid gap-6">
        {registeredModels.map((model) => (
          <Card key={model.id}>
            <CardHeader>
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    {model.name}
                  </CardTitle>
                  <CardDescription className="mt-1">{model.description}</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Badge
                    variant={
                      model.status === "Production"
                        ? "default"
                        : model.status === "Staging"
                          ? "secondary"
                          : model.status === "Development"
                            ? "outline"
                            : "destructive"
                    }
                  >
                    {model.status}
                  </Badge>
                  <span className="text-sm text-muted-foreground">{model.version}</span>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Model Metrics */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center p-3 bg-muted/50 rounded-lg">
                    <p className="text-sm text-muted-foreground">Accuracy</p>
                    <p className="text-xl font-bold text-chart-4">{model.accuracy}%</p>
                  </div>
                  <div className="text-center p-3 bg-muted/50 rounded-lg">
                    <p className="text-sm text-muted-foreground">Precision</p>
                    <p className="text-xl font-bold text-chart-1">{model.precision}%</p>
                  </div>
                  <div className="text-center p-3 bg-muted/50 rounded-lg">
                    <p className="text-sm text-muted-foreground">Recall</p>
                    <p className="text-xl font-bold text-chart-2">{model.recall}%</p>
                  </div>
                </div>

                {/* Model Info */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Model ID:</span>
                    <span className="ml-2 font-mono">{model.id}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Job ID:</span>
                    <span className="ml-2 font-mono">{model.jobId}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Registered:</span>
                    <span className="ml-2">{model.registeredAt}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Version:</span>
                    <span className="ml-2">{model.version}</span>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2 pt-2">
                  <Button variant="outline" size="sm">
                    <Eye className="w-4 h-4 mr-1" />
                    View Details
                  </Button>
                  {model.status !== "Production" && (
                    <Button size="sm">
                      <Rocket className="w-4 h-4 mr-1" />
                      Deploy
                    </Button>
                  )}
                  {model.status !== "Archived" && (
                    <Button variant="outline" size="sm">
                      <ArrowUp className="w-4 h-4 mr-1" />
                      Promote
                    </Button>
                  )}
                  <Button variant="outline" size="sm">
                    <RotateCcw className="w-4 h-4 mr-1" />
                    Rollback
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
