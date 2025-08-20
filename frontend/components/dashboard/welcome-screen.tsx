"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  Brain,
  Sparkles,
  Zap,
  GitBranch,
  TrendingUp,
  MessageSquare,
  ArrowRight,
  Play,
  Rocket,
  Target,
  Users,
  Shield,
} from "lucide-react"

interface WelcomeScreenProps {
  onGetStarted: () => void
}

export function WelcomeScreen({ onGetStarted }: WelcomeScreenProps) {
  const [animationStep, setAnimationStep] = useState(0)

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationStep(1)
    }, 500)
    return () => clearTimeout(timer)
  }, [])

  const features = [
    {
      icon: GitBranch,
      title: "No-Code Workflows",
      description: "Build ML pipelines with visual drag-and-drop interface",
      color: "from-blue-500 to-cyan-500",
    },
    {
      icon: Brain,
      title: "AI Model Registry",
      description: "Manage and version your models with intelligent tracking",
      color: "from-purple-500 to-pink-500",
    },
    {
      icon: TrendingUp,
      title: "Real-time Monitoring",
      description: "Track performance and detect drift automatically",
      color: "from-green-500 to-emerald-500",
    },
    {
      icon: MessageSquare,
      title: "AI Assistant",
      description: "Natural language interface for complex operations",
      color: "from-orange-500 to-red-500",
    },
  ]

  const stats = [
    { icon: Rocket, label: "Models Deployed", value: "2.5K+", color: "text-blue-400" },
    { icon: Target, label: "Accuracy Rate", value: "99.2%", color: "text-green-400" },
    { icon: Users, label: "Active Users", value: "150+", color: "text-purple-400" },
    { icon: Shield, label: "Uptime", value: "99.9%", color: "text-cyan-400" },
  ]

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-gradient-to-r from-primary/20 to-accent/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-r from-accent/10 to-primary/10 rounded-full blur-3xl animate-pulse delay-1000" />
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-conic from-primary/5 via-accent/5 to-primary/5 rounded-full blur-3xl animate-spin"
          style={{ animationDuration: "20s" }}
        />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-8">
        {/* Hero Section */}
        <div
          className={`text-center mb-16 transition-all duration-1000 ${animationStep >= 1 ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}
        >
          <div className="flex items-center justify-center gap-4 mb-8">
            <div className="relative group">
              <div className="w-20 h-20 bg-gradient-primary rounded-2xl flex items-center justify-center glow-primary hover:scale-110 hover:rotate-6 transition-all duration-500 cursor-pointer">
                <Brain className="w-10 h-10 text-white" />
              </div>
              <div className="absolute -top-2 -right-2 w-8 h-8 bg-accent rounded-full flex items-center justify-center hover:scale-125 transition-transform duration-300">
                <Sparkles className="w-4 h-4 text-accent-foreground animate-pulse" />
              </div>
            </div>
          </div>

          <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-foreground via-primary to-accent bg-clip-text text-transparent leading-tight">
            Welcome to
            <br />
            <span className="text-primary">Inferix</span>
          </h1>

          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
            Accelerate AI inference and model deployment with intelligent automation. Build, train, and deploy machine
            learning models with our no-code platform that transforms complex ML operations into simple workflows.
          </p>

          <div className="flex items-center justify-center gap-4">
            <Button
              onClick={onGetStarted}
              size="lg"
              className="group bg-gradient-primary text-white hover:shadow-lg hover:shadow-primary/25 hover:scale-105 transition-all duration-300 px-8 py-6 text-lg"
            >
              <Play className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform duration-300" />
              Get Started
              <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform duration-300" />
            </Button>

            <Button
              variant="outline"
              size="lg"
              className="group border-primary/50 text-foreground hover:bg-primary/10 hover:border-primary hover:text-primary hover:scale-105 transition-all duration-300 px-8 py-6 text-lg bg-transparent"
            >
              <Zap className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform duration-300" />
              Watch Demo
            </Button>
          </div>
        </div>

        {/* Stats Section */}
        <div
          className={`grid grid-cols-4 gap-6 mb-16 transition-all duration-1000 delay-300 ${animationStep >= 1 ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}
        >
          {stats.map((stat, index) => {
            const Icon = stat.icon
            return (
              <Card
                key={index}
                className="p-6 text-center glass-card hover:shadow-lg hover:shadow-primary/10 hover:scale-105 hover:-translate-y-2 transition-all duration-300 group cursor-pointer"
              >
                <Icon
                  className={`w-8 h-8 mx-auto mb-3 ${stat.color} group-hover:scale-110 transition-transform duration-300`}
                />
                <div className="text-2xl font-bold text-foreground mb-1 group-hover:text-primary transition-colors duration-300">
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </Card>
            )
          })}
        </div>

        {/* Features Grid */}
        <div
          className={`grid grid-cols-2 gap-8 transition-all duration-1000 delay-500 ${animationStep >= 1 ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}
        >
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <Card
                key={index}
                className="p-8 glass-card hover:shadow-xl hover:shadow-primary/20 hover:scale-105 hover:-translate-y-4 transition-all duration-500 group cursor-pointer relative overflow-hidden"
              >
                {/* Animated background gradient */}
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-10 transition-opacity duration-500`}
                />

                <div className="relative z-10">
                  <div
                    className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6 group-hover:scale-110 group-hover:rotate-3 transition-all duration-300`}
                  >
                    <Icon className="w-8 h-8 text-white" />
                  </div>

                  <h3 className="text-xl font-semibold mb-3 text-foreground group-hover:text-primary transition-colors duration-300">
                    {feature.title}
                  </h3>

                  <p className="text-muted-foreground leading-relaxed group-hover:text-foreground transition-colors duration-300">
                    {feature.description}
                  </p>
                </div>
              </Card>
            )
          })}
        </div>
      </div>
    </div>
  )
}
