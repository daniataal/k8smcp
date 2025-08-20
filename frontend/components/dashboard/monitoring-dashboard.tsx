import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Activity, AlertTriangle, TrendingUp, Zap, Clock, CheckCircle } from "lucide-react"

const inferenceMetrics = [
  { label: "Total Requests", value: "1.2M", change: "+8%", trend: "up" },
  { label: "Avg Latency", value: "45ms", change: "-12%", trend: "down" },
  { label: "Error Rate", value: "0.02%", change: "+0.01%", trend: "up" },
  { label: "Throughput", value: "2.3K/s", change: "+15%", trend: "up" },
]

const modelHealth = [
  { name: "MNIST CNN", status: "Healthy", uptime: 99.9, requests: "450K", latency: "32ms" },
  { name: "Sentiment BERT", status: "Healthy", uptime: 99.7, requests: "320K", latency: "78ms" },
  { name: "Image ResNet", status: "Warning", uptime: 98.2, requests: "180K", latency: "156ms" },
]

const driftAlerts = [
  {
    id: "alert-001",
    type: "Data Drift",
    model: "MNIST CNN",
    severity: "Medium",
    message: "Input distribution shift detected",
    timestamp: "2024-01-15 14:30",
    confidence: 0.73,
  },
  {
    id: "alert-002",
    type: "Performance Drift",
    model: "Sentiment BERT",
    severity: "Low",
    message: "Accuracy dropped below threshold",
    timestamp: "2024-01-15 12:15",
    confidence: 0.68,
  },
  {
    id: "alert-003",
    type: "Concept Drift",
    model: "Image ResNet",
    severity: "High",
    message: "Model predictions becoming unreliable",
    timestamp: "2024-01-15 10:45",
    confidence: 0.89,
  },
]

export function MonitoringDashboard() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold mb-2">Monitoring & Alerts</h2>
        <p className="text-muted-foreground">Real-time monitoring of model performance and system health</p>
      </div>

      {/* Inference Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {inferenceMetrics.map((metric) => (
          <Card key={metric.label}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">{metric.label}</p>
                  <p className="text-2xl font-bold">{metric.value}</p>
                  <div className="flex items-center gap-1 mt-1">
                    <TrendingUp className={`w-3 h-3 ${metric.trend === "up" ? "text-chart-4" : "text-chart-5"}`} />
                    <span className={`text-xs ${metric.trend === "up" ? "text-chart-4" : "text-chart-5"}`}>
                      {metric.change}
                    </span>
                  </div>
                </div>
                <Activity className="w-8 h-8 text-primary" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Health Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Model Health Status
            </CardTitle>
            <CardDescription>Real-time health monitoring for deployed models</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {modelHealth.map((model) => (
                <div key={model.name} className="p-4 border rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold">{model.name}</h4>
                    <Badge variant={model.status === "Healthy" ? "secondary" : "destructive"}>
                      {model.status === "Healthy" ? (
                        <CheckCircle className="w-3 h-3 mr-1" />
                      ) : (
                        <AlertTriangle className="w-3 h-3 mr-1" />
                      )}
                      {model.status}
                    </Badge>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Uptime</span>
                      <span className="font-medium">{model.uptime}%</span>
                    </div>
                    <Progress value={model.uptime} className="h-2" />
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Requests:</span>
                      <span className="ml-2 font-medium">{model.requests}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Latency:</span>
                      <span className="ml-2 font-medium">{model.latency}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Drift Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Drift Alerts
            </CardTitle>
            <CardDescription>Detected data and concept drift alerts</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {driftAlerts.map((alert) => (
                <div key={alert.id} className="p-4 border rounded-lg space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge
                        variant={
                          alert.severity === "High"
                            ? "destructive"
                            : alert.severity === "Medium"
                              ? "default"
                              : "secondary"
                        }
                      >
                        {alert.severity}
                      </Badge>
                      <span className="font-medium">{alert.type}</span>
                    </div>
                    <div className="flex items-center gap-1 text-sm text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      {alert.timestamp}
                    </div>
                  </div>

                  <div>
                    <p className="font-medium text-sm">{alert.model}</p>
                    <p className="text-sm text-muted-foreground">{alert.message}</p>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="text-sm">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="ml-2 font-medium">{(alert.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <button className="text-sm text-primary hover:underline">Acknowledge</button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Performance Chart Placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>System Performance Overview</CardTitle>
          <CardDescription>Real-time metrics and historical trends</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 bg-muted/20 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <TrendingUp className="w-12 h-12 text-muted-foreground mx-auto mb-2" />
              <p className="text-muted-foreground">Interactive charts would be displayed here</p>
              <p className="text-sm text-muted-foreground">Grafana dashboards or custom visualizations</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
