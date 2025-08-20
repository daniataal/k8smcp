"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Activity, CheckCircle, XCircle, Clock, Zap, TrendingUp, Brain, Sparkles, ArrowUpRight } from "lucide-react"
import React, { useEffect, useState } from 'react';

const workflowStats = [
  { label: "Running", count: 3, color: "bg-chart-3", icon: Clock },
  { label: "Succeeded", count: 24, color: "bg-chart-4", icon: CheckCircle },
  { label: "Failed", count: 2, color: "bg-chart-5", icon: XCircle },
]

const modelStats = [
  { name: "MNIST CNN", status: "Production", accuracy: 98.5, version: "v2.1" },
  { name: "Sentiment Analysis", status: "Staging", accuracy: 94.2, version: "v1.3" },
  { name: "Image Classifier", status: "Development", accuracy: 91.8, version: "v0.9" },
]

const recentActivity = [
  { action: "Model deployed", model: "MNIST CNN v2.1", time: "2 minutes ago", status: "success" },
  { action: "Workflow completed", model: "Sentiment Analysis", time: "15 minutes ago", status: "success" },
  { action: "Training started", model: "Image Classifier", time: "1 hour ago", status: "running" },
  { action: "Drift alert", model: "MNIST CNN", time: "3 hours ago", status: "warning" },
]

export function DashboardOverview() {
  const [healthStatus, setHealthStatus] = useState('unknown');
  const [modelName, setModelName] = useState('N/A');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await fetch('/api/health');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setHealthStatus(data.status);
        setModelName(data.model);
      } catch (e: any) {
        console.error("Failed to fetch health status:", e);
        setError(e.message);
        setHealthStatus('unhealthy'); // Set to unhealthy on error
      } finally {
        setIsLoading(false);
      }
    };

    fetchHealth();
  }, []);

  const systemHealthContent = () => {
    if (isLoading) {
      return (
        <div className="text-sm text-muted-foreground flex items-center gap-1">
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" />
          <span>Loading...</span>
        </div>
      );
    }

    if (error) {
      return (
        <div className="text-sm text-red-500 flex items-center gap-1">
          <XCircle className="w-4 h-4" />
          <span>Error: {error}</span>
        </div>
      );
    }

    const isHealthy = healthStatus === 'healthy';
    const healthColorClass = isHealthy ? 'bg-green-400' : 'bg-red-500';
    const healthText = isHealthy ? 'Operational' : 'Degraded';

    return (
      <div className="flex items-center gap-1 text-sm">
        <div className={`w-2 h-2 ${healthColorClass} rounded-full animate-pulse`} />
        <span className="text-muted-foreground">{healthText} ({modelName})</span>
      </div>
    );
  };

  // Filter and display only the MNIST CNN model if its status is 'Production' from backend
  const filteredModelStats = modelStats.filter(model => 
    model.name === modelName && healthStatus === 'healthy' && model.status === 'Production'
  );

  return (
    <div className="space-y-8">
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/20 via-accent/10 to-transparent rounded-2xl blur-3xl" />
        <div className="relative">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-primary rounded-2xl flex items-center justify-center glow-primary">
              <Brain className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-foreground via-primary to-accent bg-clip-text text-transparent">
                MLOps Dashboard
              </h1>
              <p className="text-lg text-muted-foreground">Unleash the power of intelligent ML operations</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="glass-card border-primary/20 hover:border-primary/40 transition-all duration-300 group">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Models</CardTitle>
            <div className="w-10 h-10 bg-primary/10 rounded-xl flex items-center justify-center group-hover:bg-primary/20 transition-colors">
              <Brain className="h-5 w-5 text-primary" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-1">12</div>
            <div className="flex items-center gap-1 text-sm">
              <ArrowUpRight className="w-4 h-4 text-green-400" />
              <span className="text-green-400 font-medium">+2</span>
              <span className="text-muted-foreground">from last month</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-accent/20 hover:border-accent/40 transition-all duration-300 group">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Active Workflows</CardTitle>
            <div className="w-10 h-10 bg-accent/10 rounded-xl flex items-center justify-center group-hover:bg-accent/20 transition-colors">
              <Activity className="h-5 w-5 text-accent" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-1">29</div>
            <div className="flex items-center gap-1 text-sm">
              <ArrowUpRight className="w-4 h-4 text-green-400" />
              <span className="text-green-400 font-medium">+12%</span>
              <span className="text-muted-foreground">from last week</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-chart-4/20 hover:border-chart-4/40 transition-all duration-300 group">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Inference Requests</CardTitle>
            <div className="w-10 h-10 bg-chart-4/10 rounded-xl flex items-center justify-center group-hover:bg-chart-4/20 transition-colors">
              <TrendingUp className="h-5 w-5 text-chart-4" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-1">1.2M</div>
            <div className="flex items-center gap-1 text-sm">
              <ArrowUpRight className="w-4 h-4 text-green-400" />
              <span className="text-green-400 font-medium">+8%</span>
              <span className="text-muted-foreground">from yesterday</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-chart-1/20 hover:border-chart-1/40 transition-all duration-300 group">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">System Health</CardTitle>
            <div className="w-10 h-10 bg-chart-1/10 rounded-xl flex items-center justify-center group-hover:bg-chart-1/20 transition-colors">
              <Zap className="h-5 w-5 text-chart-1" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-1">99.8%</div>
            {systemHealthContent()}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="glass-card border-primary/10">
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                <Activity className="w-4 h-4 text-primary" />
              </div>
              <div>
                <CardTitle className="text-xl">Workflow Status</CardTitle>
                <CardDescription>Current status of your ML workflows</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {workflowStats.map((stat) => {
                const Icon = stat.icon
                return (
                  <div
                    key={stat.label}
                    className="flex items-center justify-between p-4 rounded-xl bg-muted/30 hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div className={`w-4 h-4 rounded-full ${stat.color} shadow-lg`} />
                      <span className="font-medium text-lg">{stat.label}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <Icon className="w-5 h-5 text-muted-foreground" />
                      <span className="font-bold text-2xl">{stat.count}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-accent/10">
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-accent/10 rounded-lg flex items-center justify-center">
                <Brain className="w-4 h-4 text-accent" />
              </div>
              <div>
                <CardTitle className="text-xl">Model Performance</CardTitle>
                <CardDescription>Key metrics for deployed models</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {filteredModelStats.map((model) => (
                <div
                  key={model.name}
                  className="p-4 rounded-xl bg-muted/30 hover:bg-muted/50 transition-colors space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-lg">{model.name}</p>
                      <div className="flex items-center gap-3 mt-1">
                        <Badge
                          variant={model.status === "Production" ? "default" : "secondary"}
                          className={model.status === "Production" ? "bg-primary text-primary-foreground" : ""}
                        >
                          {model.status}
                        </Badge>
                        <span className="text-sm text-muted-foreground font-mono">{model.version}</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-bold text-2xl">{model.accuracy}%</p>
                      <p className="text-sm text-muted-foreground">Accuracy</p>
                    </div>
                  </div>
                  <Progress value={model.accuracy} className="h-3" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="glass-card border-muted/20">
        <CardHeader className="pb-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-muted/20 rounded-lg flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-foreground" />
            </div>
            <div>
              <CardTitle className="text-xl">Recent Activity</CardTitle>
              <CardDescription>Latest updates from your MLOps pipeline</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recentActivity.map((activity, index) => (
              <div
                key={index}
                className="flex items-center gap-4 p-4 rounded-xl bg-muted/30 hover:bg-muted/50 transition-all duration-200 hover:translate-x-1"
              >
                <div
                  className={`w-3 h-3 rounded-full shadow-lg ${
                    activity.status === "success"
                      ? "bg-chart-4 shadow-chart-4/50"
                      : activity.status === "warning"
                        ? "bg-chart-1 shadow-chart-1/50"
                        : "bg-chart-3 shadow-chart-3/50"
                  }`}
                />
                <div className="flex-1">
                  <p className="font-semibold text-lg">{activity.action}</p>
                  <p className="text-muted-foreground">{activity.model}</p>
                </div>
                <span className="text-sm text-muted-foreground font-mono bg-muted/50 px-3 py-1 rounded-lg">
                  {activity.time}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
