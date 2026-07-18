import React, { useState } from 'react';
import { MessageSquare, LayoutDashboard, Settings, Activity } from 'lucide-react';
import Chat from './Chat';
import './index.css'; // Make sure styles are loaded

function App() {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="brand">
          <Activity size={24} color="#3b82f6" />
          <span>Llama3 FinTuner</span>
        </div>
        
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <a 
            href="#" 
            className={`nav-link ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={(e) => { e.preventDefault(); setActiveTab('chat'); }}
          >
            <MessageSquare size={18} />
            Chat & Inference
          </a>
          <a 
            href="#" 
            className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={(e) => { e.preventDefault(); setActiveTab('dashboard'); }}
          >
            <LayoutDashboard size={18} />
            Training Dashboard
          </a>
          <a 
            href="#" 
            className={`nav-link ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={(e) => { e.preventDefault(); setActiveTab('settings'); }}
          >
            <Settings size={18} />
            Settings
          </a>
        </nav>
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        <div className="topbar">
          <div style={{ fontWeight: 600 }}>Enterprise AI Platform</div>
        </div>
        
        <div className="page-content">
          {activeTab === 'chat' && <Chat />}
          {activeTab === 'dashboard' && (
            <div>
              <h2 className="page-title">Training Dashboard</h2>
              <div className="card">
                <div className="card-title">Recent Experiments</div>
                <p style={{ color: 'var(--text-muted)' }}>Metrics and evaluation scores will appear here. Integrates with Weights & Biases.</p>
              </div>
            </div>
          )}
          {activeTab === 'settings' && (
            <div>
              <h2 className="page-title">Settings</h2>
              <div className="card">
                <div className="card-title">API Configuration</div>
                <p style={{ color: 'var(--text-muted)' }}>Configure endpoints, Hyperparameters for inference, etc.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
