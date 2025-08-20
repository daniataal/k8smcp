"use client"

import { useState } from "react"
import { BarChart3, Brain, GitBranch, MessageSquare, Settings, TrendingUp, Sparkles, Zap, FileUp } from "lucide-react"
import { DashboardOverview } from "./dashboard/dashboard-overview"
import { WorkflowManagement } from "./dashboard/workflow-management"
import { ModelRegistry } from "./dashboard/model-registry"
import { MonitoringDashboard } from "./dashboard/monitoring-dashboard"
import { ChatbotInterface } from "./dashboard/chatbot-interface"
import { SettingsPanel } from "./dashboard/settings-panel"
import { WelcomeScreen } from "./dashboard/welcome-screen"
import { ModelInference } from "./dashboard/model-inference";
import { KubernetesYamlUploader } from "./dashboard/kubernetes-yaml-uploader";

const navigationItems = [
  { id: "overview", label: "Dashboard", icon: BarChart3 },
  { id: "workflows", label: "Workflows", icon: GitBranch },
  { id: "models", label: "Model Registry", icon: Brain },
  { id: "inference", label: "Model Inference", icon: Sparkles },
  { id: "deploy-yaml", label: "Deploy YAML", icon: FileUp },
  { id: "monitoring", label: "Monitoring", icon: TrendingUp },
  { id: "chat", label: "AI Assistant", icon: MessageSquare },
  { id: "settings", label: "Settings", icon: Settings },
]

export function MLOpsDashboard() {
  const [activeTab, setActiveTab] = useState("welcome")
  const [showWelcome, setShowWelcome] = useState(true)

  const handleGetStarted = () => {
    setShowWelcome(false)
    setActiveTab("overview")
  }

  const handleNavigation = (tabId: string) => {
    setActiveTab(tabId)
    setShowWelcome(false)
  }

  const handleLogoClick = () => {
    setShowWelcome(true)
    setActiveTab("welcome")
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="flex">
        <div className="w-72 bg-sidebar border-r border-sidebar-border relative">
          {/* Gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />

          <div className="relative p-8">
            <div className="flex items-center gap-3 mb-12 cursor-pointer" onClick={handleLogoClick}>
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-primary rounded-xl flex items-center justify-center glow-primary hover:scale-110 hover:rotate-3 transition-all duration-300">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-accent rounded-full flex items-center justify-center hover:scale-125 transition-transform duration-200">
                  <Sparkles className="w-2.5 h-2.5 text-accent-foreground" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-sidebar-foreground tracking-tight">Inferix</h1>
                <p className="text-sm text-muted-foreground">AI Platform</p>
              </div>
            </div>

            <nav className="space-y-3">
              {navigationItems.map((item) => {
                const Icon = item.icon
                const isActive = activeTab === item.id
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavigation(item.id)}
                    className={`group w-full flex items-center gap-4 px-4 py-3.5 rounded-xl text-left transition-all duration-300 relative overflow-hidden ${
                      isActive
                        ? "bg-primary text-primary-foreground shadow-lg glow-primary scale-105"
                        : "text-sidebar-foreground hover:bg-gradient-to-r hover:from-primary/10 hover:to-accent/10 hover:text-sidebar-accent-foreground hover:translate-x-2 hover:shadow-lg hover:shadow-primary/20"
                    }`}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-accent/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                    {isActive && (
                      <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-white rounded-r-full animate-pulse" />
                    )}
                    <Icon
                      className={`w-5 h-5 transition-all duration-300 relative z-10 ${
                        isActive ? "scale-110 drop-shadow-sm" : "group-hover:scale-110 group-hover:text-primary"
                      }`}
                    />
                    <span className="font-medium relative z-10">{item.label}</span>
                    {isActive && <Zap className="w-4 h-4 ml-auto opacity-70 animate-pulse relative z-10" />}
                  </button>
                )
              })}
            </nav>

            <div className="mt-12 p-4 rounded-xl glass-card hover:shadow-lg hover:shadow-primary/10 hover:scale-105 transition-all duration-300 cursor-pointer group">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse group-hover:scale-150 transition-transform duration-300" />
                <div>
                  <p className="text-sm font-medium text-sidebar-foreground group-hover:text-primary transition-colors duration-300">
                    System Status
                  </p>
                  <p className="text-xs text-muted-foreground">All systems operational</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 relative">
          {/* Background pattern */}
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(139,92,246,0.1),transparent_50%)] pointer-events-none" />

          <div className="relative p-8">
            {showWelcome && <WelcomeScreen onGetStarted={handleGetStarted} />}
            {!showWelcome && (
              <>
                {activeTab === "overview" && <DashboardOverview />}
                {activeTab === "workflows" && <WorkflowManagement />}
                {activeTab === "models" && <ModelRegistry />}
                {activeTab === "monitoring" && <MonitoringDashboard />}
                {activeTab === "chat" && <ChatbotInterface />}
                {activeTab === "settings" && <SettingsPanel />}
                {activeTab === "inference" && <ModelInference />}
                {activeTab === "deploy-yaml" && <KubernetesYamlUploader />}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
