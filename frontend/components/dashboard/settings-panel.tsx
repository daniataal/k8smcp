"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Key, Database, Cloud, Monitor, Save, CheckCircle, XCircle, AlertTriangle } from "lucide-react"

const integrationStatus = [
  { name: "Kubernetes", status: "connected", version: "v1.28.0" },
  { name: "Prometheus", status: "connected", version: "v2.45.0" },
  { name: "Grafana", status: "disconnected", version: "v10.0.0" },
  { name: "Elasticsearch", status: "warning", version: "v8.8.0" },
]

export function SettingsPanel() {
  const [claudeApiKey, setClaudeApiKey] = useState("")
  const [openaiApiKey, setOpenaiApiKey] = useState("")
  const [kubeconfig, setKubeconfig] = useState("")
  const [containerRegistry, setContainerRegistry] = useState("")
  const [defaultNamespace, setDefaultNamespace] = useState("default")

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold mb-2">Settings</h2>
        <p className="text-muted-foreground">Configure your MLOps platform integrations and preferences</p>
      </div>

      <Tabs defaultValue="api-keys" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4 bg-muted/50">
          <TabsTrigger
            value="api-keys"
            className="hover:bg-primary/10 hover:text-primary hover:scale-105 transition-all duration-300 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
          >
            API Keys
          </TabsTrigger>
          <TabsTrigger
            value="kubernetes"
            className="hover:bg-primary/10 hover:text-primary hover:scale-105 transition-all duration-300 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
          >
            Kubernetes
          </TabsTrigger>
          <TabsTrigger
            value="monitoring"
            className="hover:bg-primary/10 hover:text-primary hover:scale-105 transition-all duration-300 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
          >
            Monitoring
          </TabsTrigger>
          <TabsTrigger
            value="integrations"
            className="hover:bg-primary/10 hover:text-primary hover:scale-105 transition-all duration-300 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
          >
            Integrations
          </TabsTrigger>
        </TabsList>

        <TabsContent value="api-keys" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="w-5 h-5" />
                LLM API Configuration
              </CardTitle>
              <CardDescription>Configure API keys for large language model integrations</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="claude-key">Claude API Key</Label>
                <Input
                  id="claude-key"
                  type="password"
                  placeholder="sk-ant-..."
                  value={claudeApiKey}
                  onChange={(e) => setClaudeApiKey(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">
                  Used for natural language processing and chatbot functionality
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="openai-key">OpenAI API Key</Label>
                <Input
                  id="openai-key"
                  type="password"
                  placeholder="sk-..."
                  value={openaiApiKey}
                  onChange={(e) => setOpenaiApiKey(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">Alternative LLM provider for assistant functionality</p>
              </div>

              <Button className="flex items-center gap-2">
                <Save className="w-4 h-4" />
                Save API Keys
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="kubernetes" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5" />
                Kubernetes Configuration
              </CardTitle>
              <CardDescription>Configure connection to your Kubernetes cluster</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="kubeconfig">Kubeconfig Path</Label>
                <Input
                  id="kubeconfig"
                  placeholder="/path/to/kubeconfig or paste config content"
                  value={kubeconfig}
                  onChange={(e) => setKubeconfig(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">
                  Path to kubeconfig file or paste the configuration content directly
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="namespace">Default Namespace</Label>
                <Input
                  id="namespace"
                  placeholder="default"
                  value={defaultNamespace}
                  onChange={(e) => setDefaultNamespace(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="registry">Container Registry</Label>
                <Input
                  id="registry"
                  placeholder="docker.io/myorg or gcr.io/project-id"
                  value={containerRegistry}
                  onChange={(e) => setContainerRegistry(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">Registry for storing and pulling ML model containers</p>
              </div>

              <Button className="flex items-center gap-2">
                <Save className="w-4 h-4" />
                Save Configuration
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Monitor className="w-5 h-5" />
                Observability Stack Deployment
              </CardTitle>
              <CardDescription>Deploy and configure monitoring and logging infrastructure</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Monitoring Stack */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Monitoring Stack</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">Prometheus</h4>
                      <Badge variant="secondary">Ready</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">Metrics collection and alerting</p>
                    <Button size="sm" className="w-full">
                      Deploy Prometheus
                    </Button>
                  </div>

                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">Grafana</h4>
                      <Badge variant="outline">Not Deployed</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">Metrics visualization and dashboards</p>
                    <Button size="sm" variant="outline" className="w-full bg-transparent">
                      Deploy Grafana
                    </Button>
                  </div>
                </div>
              </div>

              {/* Logging Stack */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Logging Stack</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">Elasticsearch</h4>
                      <Badge variant="destructive">Error</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">Log storage and search</p>
                    <Button size="sm" variant="outline" className="w-full bg-transparent">
                      Deploy Elasticsearch
                    </Button>
                  </div>

                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">Kibana</h4>
                      <Badge variant="outline">Not Deployed</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">Log visualization and analysis</p>
                    <Button size="sm" variant="outline" className="w-full bg-transparent">
                      Deploy Kibana
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="integrations" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cloud className="w-5 h-5" />
                Integration Status
              </CardTitle>
              <CardDescription>Monitor the status of your platform integrations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {integrationStatus.map((integration) => (
                  <div key={integration.name} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          integration.status === "connected"
                            ? "bg-chart-4"
                            : integration.status === "warning"
                              ? "bg-chart-1"
                              : "bg-chart-5"
                        }`}
                      />
                      <div>
                        <p className="font-medium">{integration.name}</p>
                        <p className="text-sm text-muted-foreground">{integration.version}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {integration.status === "connected" && <CheckCircle className="w-4 h-4 text-chart-4" />}
                      {integration.status === "warning" && <AlertTriangle className="w-4 h-4 text-chart-1" />}
                      {integration.status === "disconnected" && <XCircle className="w-4 h-4 text-chart-5" />}
                      <Badge
                        variant={
                          integration.status === "connected"
                            ? "secondary"
                            : integration.status === "warning"
                              ? "default"
                              : "destructive"
                        }
                      >
                        {integration.status}
                      </Badge>
                    </div>
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
