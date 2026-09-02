import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Bot,
  Check,
  ChevronDown,
  ChevronRight,
  CirclePlus,
  Copy,
  Database,
  FileText,
  Menu,
  PanelLeftClose,
  Search,
  Sparkles,
  Trash2,
  User,
} from "lucide-react";
import "./App.css";

const API_URL = "https://production-agentic-rag.onrender.com";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [detailsOpen, setDetailsOpen] = useState({});
  const [customEnabled, setCustomEnabled] = useState(false);
  const [customStatus, setCustomStatus] = useState("Default Handbook");
  const [customKnowledge, setCustomKnowledge] = useState({ handbook: [], qa: [] });
  const [qaForm, setQaForm] = useState({ question: "", answer: "" });
  const [uploadName, setUploadName] = useState("");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  useEffect(() => {
    const loadCustomKnowledge = async () => {
      try {
        const response = await fetch(`${API_URL}/knowledge/custom`);
        if (!response.ok) return;
        const data = await response.json();
        setCustomKnowledge(data);
        setCustomEnabled(Boolean(data.enabled));
        setCustomStatus(
          data.enabled
            ? data.qa?.length || data.handbook?.length
              ? "Custom + Default fallback"
              : "Custom Knowledge"
            : "Default Handbook"
        );
      } catch (error) {
        console.error("Failed to load custom knowledge", error);
      }
    };

    loadCustomKnowledge();
  }, []);

  const suggestedQuestions = [
    "What is the return policy?",
    "How long do refunds take?",
    "What are the shipping options?",
  ];

  const askQuestion = async (customQuestion = null) => {
    const userQuestion = (customQuestion || question).trim();

    if (!userQuestion || loading) return;

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: userQuestion,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: userQuestion,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations || [],
          faithfulness: data.faithfulness,
          iterations: data.iterations,
          blocked: data.blocked,
        },
      ]);
    } catch (error) {
      console.error("API error:", error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "I couldn't connect to the RAG server. The backend may be waking up or temporarily unavailable. Please try again in a moment.",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion();
    }
  };

  const toggleCustomKnowledge = async () => {
    const nextValue = !customEnabled;
    setCustomEnabled(nextValue);
    setCustomStatus(nextValue ? "Custom Knowledge" : "Default Handbook");

    try {
      await fetch(`${API_URL}/knowledge/custom/enable`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: nextValue }),
      });
      const response = await fetch(`${API_URL}/knowledge/custom`);
      if (response.ok) {
        const data = await response.json();
        setCustomKnowledge(data);
        setCustomStatus(
          data.enabled
            ? data.qa?.length || data.handbook?.length
              ? "Custom + Default fallback"
              : "Custom Knowledge"
            : "Default Handbook"
        );
      }
    } catch (error) {
      console.error("Failed to update custom knowledge", error);
      setCustomEnabled(!nextValue);
    }
  };

  const uploadCustomHandbook = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/knowledge/custom/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Upload failed");
      const data = await response.json();
      setUploadName(data.name || file.name);
      const detail = await fetch(`${API_URL}/knowledge/custom`);
      if (detail.ok) {
        const knowledge = await detail.json();
        setCustomKnowledge(knowledge);
      }
    } catch (error) {
      console.error("Custom handbook upload failed", error);
    }
  };

  const addCustomQA = async () => {
    if (!qaForm.question.trim() || !qaForm.answer.trim()) return;

    try {
      await fetch(`${API_URL}/knowledge/custom/qa`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: qaForm.question,
          answer: qaForm.answer,
        }),
      });
      setQaForm({ question: "", answer: "" });
      const response = await fetch(`${API_URL}/knowledge/custom`);
      if (response.ok) {
        const data = await response.json();
        setCustomKnowledge(data);
      }
    } catch (error) {
      console.error("Failed to add custom Q&A", error);
    }
  };

  const removeCustomKnowledge = async () => {
    try {
      await fetch(`${API_URL}/knowledge/custom`, { method: "DELETE" });
      setCustomKnowledge({ handbook: [], qa: [] });
      setCustomEnabled(false);
      setCustomStatus("Default Handbook");
    } catch (error) {
      console.error("Failed to remove custom knowledge", error);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setQuestion("");
    setDetailsOpen({});
  };

  const copyMessage = async (content, index) => {
    try {
      await navigator.clipboard.writeText(content);

      setDetailsOpen((prev) => ({
        ...prev,
        [`copied-${index}`]: true,
      }));

      setTimeout(() => {
        setDetailsOpen((prev) => ({
          ...prev,
          [`copied-${index}`]: false,
        }));
      }, 1500);
    } catch (error) {
      console.error("Copy failed:", error);
    }
  };

  const toggleDetails = (index) => {
    setDetailsOpen((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "collapsed"}`}>
        <div className="sidebar-inner">
          <div className="brand-row">
            <div className="brand">
              <div className="brand-icon">
                <Sparkles size={18} />
              </div>

              {sidebarOpen && (
                <div className="brand-text">
                  <span className="brand-name">Agentic RAG</span>
                  <span className="brand-subtitle">
                    Knowledge Intelligence
                  </span>
                </div>
              )}
            </div>

            {sidebarOpen && (
              <button
                className="icon-button sidebar-toggle"
                onClick={() => setSidebarOpen(false)}
                title="Collapse sidebar"
              >
                <PanelLeftClose size={18} />
              </button>
            )}
          </div>

          <button className="new-chat-button" onClick={clearChat}>
            <CirclePlus size={19} />
            {sidebarOpen && <span>New conversation</span>}
          </button>

          {sidebarOpen && (
            <>
              <div className="sidebar-section">
                <span className="sidebar-label">WORKSPACE</span>

                <div className="workspace-card">
                  <div className="workspace-icon">
                    <Database size={17} />
                  </div>

                  <div>
                    <strong>Knowledge Base</strong>
                    <span>Connected & ready</span>
                  </div>

                  <Check size={15} className="workspace-check" />
                </div>
              </div>

              <div className="sidebar-section conversation-section">
                <span className="sidebar-label">CUSTOM KNOWLEDGE</span>

                <div className="custom-knowledge-panel">
                  <div className="custom-knowledge-header">
                    <span>Knowledge Source</span>
                    <button
                      className={`toggle-button ${customEnabled ? "on" : ""}`}
                      onClick={toggleCustomKnowledge}
                    >
                      {customEnabled ? "Enabled" : "Disabled"}
                    </button>
                  </div>

                  <div className="source-badge">{customStatus}</div>

                  <label className="upload-box">
                    <input type="file" onChange={uploadCustomHandbook} />
                    <span>Add Custom Handbook</span>
                  </label>

                  {uploadName && (
                    <div className="file-status">{uploadName}</div>
                  )}

                  {customKnowledge.handbook?.length > 0 && (
                    <div className="knowledge-list">
                      {customKnowledge.handbook.map((item) => (
                        <div className="knowledge-item" key={item.id}>
                          <FileText size={14} />
                          <span>{item.name}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="qa-editor">
                    <label>
                      Question
                      <input
                        value={qaForm.question}
                        onChange={(event) =>
                          setQaForm((prev) => ({ ...prev, question: event.target.value }))
                        }
                        placeholder="What is our refund policy?"
                      />
                    </label>

                    <label>
                      Answer
                      <textarea
                        value={qaForm.answer}
                        onChange={(event) =>
                          setQaForm((prev) => ({ ...prev, answer: event.target.value }))
                        }
                        placeholder="Customers can request a refund within 14 days."
                        rows={3}
                      />
                    </label>

                    <button className="primary-button" onClick={addCustomQA}>
                      Add Q&A
                    </button>
                  </div>

                  {customKnowledge.qa?.length > 0 && (
                    <div className="knowledge-list">
                      {customKnowledge.qa.map((item) => (
                        <div className="qa-item" key={item.id}>
                          <div>
                            <strong>{item.question}</strong>
                            <span>{item.answer}</span>
                          </div>
                          <button
                            aria-label="Delete Q&A"
                            onClick={async () => {
                              await fetch(`${API_URL}/knowledge/custom/qa/${item.id}`, { method: "DELETE" });
                              const response = await fetch(`${API_URL}/knowledge/custom`);
                              if (response.ok) {
                                setCustomKnowledge(await response.json());
                              }
                            }}
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  <button className="secondary-button" onClick={removeCustomKnowledge}>
                    Remove Custom Handbook
                  </button>
                </div>
              </div>

              <div className="sidebar-section conversation-section">
                <span className="sidebar-label">CONVERSATION</span>

                <button className="conversation-item active">
                  <Bot size={16} />
                  <span>Current conversation</span>
                </button>
              </div>

              <div className="sidebar-footer">
                <div className="system-status-card">
                  <div className="status-indicator">
                    <span></span>
                  </div>

                  <div>
                    <strong>System operational</strong>
                    <span>RAG pipeline online</span>
                  </div>
                </div>
              </div>
            </>
          )}

          {!sidebarOpen && (
            <div className="collapsed-sidebar-actions">
              <button
                className="icon-button"
                onClick={() => setSidebarOpen(true)}
                title="Open sidebar"
              >
                <Menu size={19} />
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* Main Application */}
      <main className="main-content">
        {/* Top Bar */}
        <header className="topbar">
          <div className="topbar-left">
            {!sidebarOpen && (
              <button
                className="icon-button mobile-sidebar-button"
                onClick={() => setSidebarOpen(true)}
                title="Open sidebar"
              >
                <Menu size={20} />
              </button>
            )}

            <div className="page-title">
              <h1>Knowledge Assistant</h1>
              <span>Ask, search, understand</span>
            </div>
          </div>

          <div className="topbar-status">
            <span className="live-dot"></span>
            <span>Online</span>
          </div>
        </header>

        {/* Chat Area */}
        <section className="chat-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="hero-badge">
                <Sparkles size={20} />
              </div>

              <h2>What would you like to know?</h2>

              <p>
                Search your knowledge base with an intelligent,
                citation-aware AI assistant.
              </p>

              <div className="suggestions">
                {suggestedQuestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    className="suggestion-card"
                    onClick={() => askQuestion(suggestion)}
                    disabled={loading}
                  >
                    <div className="suggestion-icon">
                      <Search size={17} />
                    </div>

                    <span>{suggestion}</span>

                    <ChevronRight size={16} />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages-container">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`message-row ${message.role}`}
                >
                  {message.role === "assistant" && (
                    <div className="assistant-avatar">
                      <Sparkles size={17} />
                    </div>
                  )}

                  <div className="message-wrapper">
                    <div className="message-label">
                      {message.role === "assistant"
                        ? "Agentic RAG"
                        : "You"}
                    </div>

                    <div
                      className={`message-text ${
                        message.error ? "error" : ""
                      }`}
                    >
                      {message.content}
                    </div>

                    {message.role === "assistant" && !message.error && (
                      <div className="message-actions">
                        <button
                          className="message-action"
                          onClick={() =>
                            copyMessage(message.content, index)
                          }
                        >
                          {detailsOpen[`copied-${index}`] ? (
                            <>
                              <Check size={14} />
                              Copied
                            </>
                          ) : (
                            <>
                              <Copy size={14} />
                              Copy
                            </>
                          )}
                        </button>

                        {message.citations?.length > 0 && (
                          <button
                            className="message-action"
                            onClick={() => toggleDetails(index)}
                          >
                            <FileText size={14} />
                            Sources & details

                            {detailsOpen[index] ? (
                              <ChevronDown size={14} />
                            ) : (
                              <ChevronRight size={14} />
                            )}
                          </button>
                        )}
                      </div>
                    )}

                    {message.role === "assistant" &&
                      detailsOpen[index] && (
                        <div className="response-details">
                          {message.citations?.length > 0 && (
                            <div className="sources-section">
                              <span className="detail-label">
                                SOURCES
                              </span>

                              <div className="source-chips">
                                {message.citations.map((citation) => (
                                  <div
                                    className="source-chip"
                                    key={citation}
                                  >
                                    <FileText size={13} />
                                    {citation}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          <div className="quality-grid">
                            {message.faithfulness !== undefined && (
                              <div className="quality-item">
                                <span>GROUNDING</span>
                                <strong>
                                  {Math.round(
                                    message.faithfulness * 100
                                  )}
                                  %
                                </strong>
                              </div>
                            )}

                            {message.iterations !== undefined && (
                              <div className="quality-item">
                                <span>REASONING PASSES</span>
                                <strong>{message.iterations}</strong>
                              </div>
                            )}

                            {message.blocked !== undefined && (
                              <div className="quality-item">
                                <span>SAFETY</span>
                                <strong>
                                  {message.blocked
                                    ? "Blocked"
                                    : "Passed"}
                                </strong>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                  </div>

                  {message.role === "user" && (
                    <div className="user-avatar">
                      <User size={17} />
                    </div>
                  )}
                </div>
              ))}

              {loading && (
                <div className="message-row assistant">
                  <div className="assistant-avatar">
                    <Sparkles size={17} />
                  </div>

                  <div className="message-wrapper">
                    <div className="message-label">
                      Agentic RAG
                    </div>

                    <div className="thinking-state">
                      <div className="thinking-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>

                      <span>Searching knowledge base...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </section>

        {/* Input */}
        <div className="composer-wrapper">
          <div className="composer">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your knowledge base..."
              rows={1}
              disabled={loading}
            />

            <button
              className="send-button"
              onClick={() => askQuestion()}
              disabled={!question.trim() || loading}
              title="Send message"
            >
              <ArrowUp size={20} />
            </button>
          </div>

          <div className="composer-footer">
            <span>
              <kbd>Enter</kbd> to send
            </span>

            <span>
              <kbd>Shift + Enter</kbd> for a new line
            </span>

            <span className="powered-by">
              Powered by hybrid retrieval
            </span>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
