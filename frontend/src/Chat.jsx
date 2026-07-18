import React, { useState } from 'react';
import axios from 'axios';
import { Send, Cpu, SlidersHorizontal } from 'lucide-react';

export default function Chat() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Hello! I am the Llama 3 Financial Assistant. How can I help you today?', model_used: null }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useLora, setUseLora] = useState(true);

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    
    try {
      const response = await axios.post('http://localhost:8000/api/v1/chat', {
        instruction: userMsg.text,
        use_lora: useLora,
        max_tokens: 512,
        temperature: 0.7,
        top_p: 0.9
      });
      
      const assistantMsg = { 
        role: 'assistant', 
        text: response.data.response,
        model_used: response.data.model_used,
        latency: response.data.latency_ms
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages(prev => [...prev, { role: 'assistant', text: 'Sorry, I encountered an error communicating with the backend.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', gap: '1.5rem', height: '100%' }}>
      {/* Main Chat Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2 className="page-title" style={{ margin: 0 }}>Model Inference</h2>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
              Using: {useLora ? 'Fine-Tuned (LoRA)' : 'Base Llama 3'}
            </span>
          </div>
        </div>
        
        <div className="chat-container">
          <div className="chat-history">
            {messages.map((msg, i) => (
              <div key={i} style={{ display: 'flex', flexDirection: 'column' }}>
                <div className={`chat-message ${msg.role}`}>
                  {msg.text}
                </div>
                {msg.model_used && (
                  <div style={{ alignSelf: 'flex-start', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    Model: {msg.model_used} | Latency: {msg.latency?.toFixed(0)}ms
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="chat-message assistant" style={{ fontStyle: 'italic', opacity: 0.7 }}>
                Generating response...
              </div>
            )}
          </div>
          
          <div className="chat-input-container">
            <textarea 
              className="chat-input" 
              placeholder="Ask a financial question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              rows={2}
            />
            <button className="btn btn-primary" onClick={sendMessage} disabled={isLoading}>
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      <div style={{ width: '300px', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <SlidersHorizontal size={18} />
            Model Selection
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input 
                type="radio" 
                checked={useLora === true} 
                onChange={() => setUseLora(true)}
              />
              <span>Fine-Tuned (Finance)</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input 
                type="radio" 
                checked={useLora === false} 
                onChange={() => setUseLora(false)}
              />
              <span>Base Model (Zero-Shot)</span>
            </label>
          </div>
        </div>
        
        <div className="card">
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Cpu size={18} />
            Parameters
          </div>
          <div style={{ marginTop: '1rem' }}>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Temperature (0.7)</p>
            <input type="range" min="0" max="200" defaultValue="70" style={{ width: '100%', marginTop: '0.5rem' }} />
          </div>
          <div style={{ marginTop: '1rem' }}>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Max Tokens (512)</p>
            <input type="range" min="10" max="2048" defaultValue="512" style={{ width: '100%', marginTop: '0.5rem' }} />
          </div>
        </div>
      </div>
    </div>
  );
}
