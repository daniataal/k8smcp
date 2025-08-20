"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { MessageSquare, Send, Bot, User, Sparkles, Clock } from "lucide-react"

const chatHistory = [
  {
    id: 1,
    type: "user",
    message: "Show me the status of all running workflows",
    timestamp: "14:30",
  },
  {
    id: 2,
    type: "bot",
    message:
      "I found 3 running workflows:\n\n1. **MNIST Training Pipeline** (65% complete)\n2. **Image Preprocessing** (23% complete)\n3. **Model Validation** (89% complete)\n\nWould you like me to show detailed logs for any of these?",
    timestamp: "14:30",
  },
  {
    id: 3,
    type: "user",
    message: "Deploy the latest MNIST model to production",
    timestamp: "14:32",
  },
  {
    id: 4,
    type: "bot",
    message:
      "I'll deploy MNIST CNN v2.1 to production. Here's what I'm doing:\n\n✅ Validating model metrics (98.5% accuracy)\n✅ Checking resource availability\n🔄 Creating deployment configuration\n⏳ Deploying to Kubernetes cluster...\n\nDeployment initiated! Job ID: deploy-xyz789",
    timestamp: "14:32",
  },
]

const suggestedQueries = [
  "Show model performance metrics",
  "List failed workflows from today",
  "Deploy staging model to production",
  "Check system health status",
  "Create new training pipeline",
  "Show drift alerts for BERT model",
]

export function ChatbotInterface() {
  const [message, setMessage] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [currentChatHistory, setCurrentChatHistory] = useState(chatHistory); // New state for dynamic chat history

  const handleSendMessage = async () => {
    if (!message.trim()) return;

    const userMessage = { id: currentChatHistory.length + 1, type: "user", message: message, timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) };
    setCurrentChatHistory((prev) => [...prev, userMessage]);
    setMessage("");
    setIsTyping(true);

    if (message.toLowerCase() === "check health") {
      try {
        const response = await fetch('/api/health');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const botResponse = {
          id: currentChatHistory.length + 2,
          type: "bot",
          message: `Backend Health: ${data.status.toUpperCase()} (Model: ${data.model})`,
          timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        };
        setCurrentChatHistory((prev) => [...prev, botResponse]);
      } catch (e: any) {
        const errorResponse = {
          id: currentChatHistory.length + 2,
          type: "bot",
          message: `Error checking health: ${e.message}`,
          timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        };
        setCurrentChatHistory((prev) => [...prev, errorResponse]);
      } finally {
        setIsTyping(false);
      }
    } else {
      // Simulate sending message and bot response for other queries
      setTimeout(() => {
        const botResponse = {
          id: currentChatHistory.length + 2,
          type: "bot",
          message: "I'm not sure how to handle that query yet. Try 'check health'.",
          timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        };
        setCurrentChatHistory((prev) => [...prev, botResponse]);
        setIsTyping(false);
      }, 1500);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold mb-2">MLOps Assistant</h2>
        <p className="text-muted-foreground">Interact with your MLOps platform using natural language</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chat Interface */}
        <div className="lg:col-span-2">
          <Card className="h-[600px] flex flex-col">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="w-5 h-5" />
                Chat with Assistant
              </CardTitle>
              <CardDescription>
                Ask questions, execute commands, and get insights about your ML operations
              </CardDescription>
            </CardHeader>

            <CardContent className="flex-1 flex flex-col">
              {/* Chat Messages */}
              <div className="flex-1 space-y-4 overflow-y-auto mb-4">
                {currentChatHistory.map((chat) => (
                  <div key={chat.id} className={`flex gap-3 ${chat.type === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`flex gap-3 max-w-[80%] ${chat.type === "user" ? "flex-row-reverse" : "flex-row"}`}>
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          chat.type === "user" ? "bg-primary" : "bg-secondary"
                        }`}
                      >
                        {chat.type === "user" ? (
                          <User className="w-4 h-4 text-primary-foreground" />
                        ) : (
                          <Bot className="w-4 h-4 text-secondary-foreground" />
                        )}
                      </div>
                      <div
                        className={`p-3 rounded-lg ${
                          chat.type === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                        }`}
                      >
                        <div className="whitespace-pre-line text-sm">{chat.message}</div>
                        <div className={`text-xs mt-1 opacity-70`}>{chat.timestamp}</div>
                      </div>
                    </div>
                  </div>
                ))}

                {isTyping && (
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                      <Bot className="w-4 h-4 text-secondary-foreground" />
                    </div>
                    <div className="bg-muted p-3 rounded-lg">
                      <div className="flex items-center gap-1">
                        <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                        <div
                          className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                          style={{ animationDelay: "0.1s" }}
                        />
                        <div
                          className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Message Input */}
              <div className="flex gap-2">
                <Input
                  placeholder="Ask about workflows, models, or system status..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                  className="flex-1"
                />
                <Button onClick={handleSendMessage} disabled={!message.trim()}>
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar with Suggestions and Status */}
        <div className="space-y-6">
          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5" />
                Quick Actions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {suggestedQueries.map((query, index) => (
                  <button
                    key={index}
                    onClick={() => setMessage(query)}
                    className="w-full text-left p-2 text-sm rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                  >
                    {query}
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Assistant Status */}
          <Card>
            <CardHeader>
              <CardTitle>Assistant Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Connection</span>
                  <Badge variant="secondary">Online</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Response Time</span>
                  <span className="text-sm font-medium">~1.2s</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Available Tools</span>
                  <span className="text-sm font-medium">12</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Last Updated</span>
                  <div className="flex items-center gap-1 text-sm">
                    <Clock className="w-3 h-3" />2 min ago
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Available Commands */}
          <Card>
            <CardHeader>
              <CardTitle>Available Commands</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="p-2 bg-muted/50 rounded">
                  <code className="text-primary">list workflows</code>
                  <p className="text-muted-foreground mt-1">Show all workflows</p>
                </div>
                <div className="p-2 bg-muted/50 rounded">
                  <code className="text-primary">deploy model [name]</code>
                  <p className="text-muted-foreground mt-1">Deploy a model</p>
                </div>
                <div className="p-2 bg-muted/50 rounded">
                  <code className="text-primary">show metrics</code>
                  <p className="text-muted-foreground mt-1">Display performance metrics</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
