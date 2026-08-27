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
  MessageSquare,
  PanelLeftClose,
  RotateCcw,
  Search,
  Sparkles,
  User,
  X,
} from "lucide-react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [detailsOpen, setDetailsOpen] = useState({});
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

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
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: userQuestion,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
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
      console.error(error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "I couldn't connect to the RAG server. Please make sure the FastAPI backend is running.",
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
                  <span className="brand-subtitle">Knowledge Intelligence</span>
                </div>
              )}
            </div>

            <button
              className="icon-button sidebar-toggle"
              onClick={() => setSidebarOpen(false)}
              title="Collapse sidebar"
            >
              <PanelLeftClose size={18} />
            </button>
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
                <span className="sidebar-label">CONVERSATION</span>

                <button className="conversation-item active">
                  <MessageSquare size={16} />
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

                    <div className={`message-text ${message.error ? "error" : ""}`}>
                      {message.content}
                    </div>

                    {message.role === "assistant" && (
                      <div className="message-actions">
                        <button
                          className="message-action"
                          onClick={() => copyMessage(message.content, index)}
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
                              <span className="detail-label">SOURCES</span>

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
                                  {message.blocked ? "Blocked" : "Passed"}
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
                    <div className="message-label">Agentic RAG</div>

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