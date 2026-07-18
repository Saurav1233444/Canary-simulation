import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Brush } from 'recharts';
import { AlertCircle, CheckCircle2, RefreshCw, Activity, ShieldAlert, Cpu, Zap, Settings2, Sliders } from 'lucide-react';
import { getStep, resetSimulation, getHistory, injectAnomaly, setSettings, stopTraining, getTrainingInfo, uploadCSV } from './api';

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentRisk, setCurrentRisk] = useState(0);
  const [alerts, setAlerts] = useState([]);
  const [systemStatus, setSystemStatus] = useState('normal'); // normal, warning, critical
  const [datasetType, setDatasetType] = useState('sudden_shift');
  const [simulationSpeed, setSimulationSpeed] = useState(500);
  const [alertThreshold, setAlertThreshold] = useState(0.05);
  const [trainingInfo, setTrainingInfo] = useState(null);
  const [customBatchSize, setCustomBatchSize] = useState('');
  const [customEpochs, setCustomEpochs] = useState('');
  const [mlTaskType, setMlTaskType] = useState('Classification');
  const [mlModelName, setMlModelName] = useState('Logistic Regression');
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    if (datasetType === 'mobilenet_training' || datasetType === 'custom_model_training' || datasetType === 'csv_upload') {
      const targetType = datasetType === 'mobilenet_training' ? 'mobilenet_training' : 'custom_model_training';
      getTrainingInfo(targetType).then(res => {
        setTrainingInfo(res.data);
        setCustomBatchSize(res.data.predicted_batch_size);
        setCustomEpochs(res.data.predicted_epochs);
      }).catch(console.error);
    }
  }, [datasetType]);

  // Initialize data
  useEffect(() => {
    const init = async () => {
      try {
        await resetSimulation('sudden_shift');
        const res = await getHistory();
        if (res.data.alert_threshold) setAlertThreshold(res.data.alert_threshold);
        // Format history for charts
        const historyData = res.data.times.map((t, i) => ({
          time: t,
          value: res.data.values[i],
          probability: res.data.probabilities[i],
          isAlert: res.data.alerts.includes(t)
        }));
        setData(historyData);
      } catch (err) {
        console.error("Failed to fetch initial data", err);
      }
    };
    init();
    
    return () => clearInterval(timerRef.current);
  }, []);

  // Main simulation loop
  useEffect(() => {
    if (isRunning) {
      timerRef.current = setInterval(async () => {
        try {
          const res = await getStep();
          const newStep = res.data;
          
          setData(prev => {
            const newData = [...prev, newStep];
            if (newData.length > 500) return newData.slice(newData.length - 500);
            return newData;
          });
          
          setCurrentRisk(newStep.risk_score);
          
          if (newStep.is_alert) {
            setSystemStatus('critical');
            setAlerts(prev => [{
              id: Date.now(),
              time: newStep.time,
              risk: newStep.risk_score,
              status: 'pending' // pending, acknowledged, dismissed
            }, ...prev].slice(0, 10)); // Keep last 10
          } else if (newStep.risk_score > 10) {
              setSystemStatus('warning');
          } else {
             setSystemStatus('normal');
          }

        } catch (err) {
          console.error("Simulation step failed", err);
          setIsRunning(false);
        }
      }, simulationSpeed);
    } else {
      clearInterval(timerRef.current);
    }
    
    return () => clearInterval(timerRef.current);
  }, [isRunning, simulationSpeed]);

  const ML_MODELS = {
    'Classification': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'XGBoost ⭐', 'K-Nearest Neighbors', 'Naive Bayes'],
    'Regression': ['Linear Regression', 'Polynomial Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest Regressor', 'Gradient Boosting', 'XGBoost ⭐', 'Support Vector Regression'],
    'Clustering': ['K-Means ⭐', 'Hierarchical Clustering', 'DBSCAN', 'Gaussian Mixture Model', 'Mean Shift']
  };

  useEffect(() => {
     if (ML_MODELS[mlTaskType] && !ML_MODELS[mlTaskType].includes(mlModelName)) {
         setMlModelName(ML_MODELS[mlTaskType][0]);
     }
  }, [mlTaskType, mlModelName]);

  const handleFileUpload = async (event) => {
      const file = event.target.files[0];
      if (!file) return;
      setUploading(true);
      try {
          const res = await uploadCSV(file);
          if (res.data && res.data.error) {
              alert("Upload failed: " + res.data.error);
          } else {
              setDatasetType('csv_upload');
              setIsRunning(false);
              setData([]);
              setAlerts([]);
              setCurrentRisk(0);
              setSystemStatus('normal');
          }
      } catch (err) {
          alert("Error uploading file");
          console.error(err);
      } finally {
          setUploading(false);
          event.target.value = null;
      }
  };

  const handleReset = async (type = datasetType, bs = null, ep = null, taskType = null, modelName = null) => {
    setIsRunning(false);
    await resetSimulation(type, bs, ep, taskType, modelName);
    setData([]);
    setAlerts([]);
    setCurrentRisk(0);
    setSystemStatus('normal');
  };
  
  const handleAcknowledge = (id) => {
      setAlerts(prev => prev.map(a => a.id === id ? {...a, status: 'acknowledged'} : a));
      setSystemStatus('warning');
  };

  const handleDismiss = (id) => {
      setAlerts(prev => prev.map(a => a.id === id ? {...a, status: 'dismissed'} : a));
  }

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-50 p-6 md:p-10 font-sans antialiased selection:bg-zinc-800">
      
      {/* HEADER */}
      <header className="flex flex-col md:flex-row md:justify-between items-start md:items-center mb-10 pb-6 border-b border-zinc-800/60 gap-6">
        <div className="flex items-center gap-4">
          <div className="p-2.5 bg-zinc-900 border border-zinc-800 rounded-xl">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-medium tracking-tight">Canary</h1>
            <p className="text-xs text-zinc-400 font-light tracking-wide uppercase mt-0.5">Early Warning System</p>
          </div>
        </div>
        
        <div className="flex flex-wrap items-center gap-3">
          <div className={`px-4 py-1.5 rounded-full border text-xs font-medium flex items-center gap-1.5 transition-colors
              ${systemStatus === 'normal' ? 'bg-zinc-900 border-zinc-800 text-zinc-300' : ''}
              ${systemStatus === 'warning' ? 'bg-amber-950/30 border-amber-900/50 text-amber-500' : ''}
              ${systemStatus === 'critical' ? 'bg-red-950/30 border-red-900/50 text-red-500 animate-pulse' : ''}
          `}>
              {systemStatus === 'normal' && <CheckCircle2 className="w-3.5 h-3.5 opacity-70" />}
              {systemStatus === 'warning' && <AlertCircle className="w-3.5 h-3.5" />}
              {systemStatus === 'critical' && <ShieldAlert className="w-3.5 h-3.5" />}
              {systemStatus.charAt(0).toUpperCase() + systemStatus.slice(1)}
          </div>

          <select 
            value={datasetType}
            onChange={(e) => {
              const newType = e.target.value;
              setDatasetType(newType);
              handleReset(newType);
            }}
            className="bg-zinc-900 text-zinc-300 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:border-zinc-700 transition-colors"
          >
            <option value="sudden_shift">Sudden Shift</option>
            <option value="sudden_spike">Transient Spike</option>
            <option value="variance_shift">Variance Shift</option>
            <option value="gradual_drift">Continuous Drift</option>
            <option value="mobilenet_training">MobileNet Training</option>
            <option value="csv_upload">Custom Data Upload</option>
          </select>

          <select 
            value={simulationSpeed}
            onChange={(e) => setSimulationSpeed(Number(e.target.value))}
            className="bg-zinc-900 text-zinc-300 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:border-zinc-700 transition-colors hidden sm:block"
          >
            <option value={1000}>1x</option>
            <option value={500}>2x</option>
            <option value={200}>5x</option>
            <option value={50}>Max</option>
          </select>

          <div className="h-4 w-px bg-zinc-800 mx-1 hidden sm:block" />

          <button 
            onClick={async () => await injectAnomaly()}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors bg-transparent border border-transparent hover:border-zinc-700"
            title="Inject Anomaly"
          >
            <Zap className="w-4 h-4" />
          </button>

          <button 
            onClick={() => handleReset(datasetType, customBatchSize || null, customEpochs || null)}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors bg-transparent border border-transparent hover:border-zinc-700"
            title="Reset"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          <button 
            onClick={() => {
                if (datasetType === 'csv_upload' || datasetType === 'custom_model_training') {
                    setDatasetType('custom_model_training');
                    handleReset('custom_model_training', customBatchSize || null, customEpochs || null, mlTaskType, mlModelName).then(() => setIsRunning(true));
                } else {
                    setDatasetType('mobilenet_training');
                    handleReset('mobilenet_training', customBatchSize || null, customEpochs || null).then(() => setIsRunning(true));
                }
            }}
            className="ml-2 px-4 py-1.5 rounded-lg text-xs font-medium transition-all bg-zinc-800 text-white hover:bg-zinc-700 border border-zinc-700"
          >
            Train
          </button>

          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            accept=".csv, .xlsx, .xls" 
            className="hidden" 
          />
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all bg-transparent border hover:bg-zinc-800 ${uploading ? 'text-zinc-500 border-zinc-800' : 'text-zinc-300 border-zinc-700'}`}
          >
            {uploading ? 'Uploading...' : 'Upload Data'}
          </button>

          {datasetType === 'mobilenet_training' && (
            <button 
              onClick={async () => {
                  setIsRunning(false);
                  await stopTraining();
              }}
              className="px-4 py-1.5 rounded-lg text-xs font-medium transition-all bg-transparent text-red-500 hover:bg-red-950/30 border border-red-900/50"
            >
              Stop
            </button>
          )}

          <button 
            onClick={() => setIsRunning(!isRunning)}
            className={`px-5 py-1.5 rounded-lg text-xs font-medium transition-all duration-300 ${
              isRunning 
                ? 'bg-transparent text-zinc-300 border border-zinc-700 hover:bg-zinc-800' 
                : 'bg-white text-black hover:bg-zinc-200 border border-transparent'
            }`}
          >
            {isRunning ? 'Pause' : 'Start Monitoring'}
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
        
        {/* MAIN AREA */}
        <div className="xl:col-span-3 space-y-8">
          
          {/* STATS */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
              <div className="bg-zinc-900/40 border-[0.5px] border-zinc-800/60 rounded-2xl p-6 transition-all hover:bg-zinc-900/60 flex flex-col justify-between">
                  <div className="flex justify-between items-center mb-4">
                    <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Risk Score</p>
                    <ShieldAlert className={`w-4 h-4 ${currentRisk > 30 ? 'text-red-500' : 'text-zinc-600'}`} />
                  </div>
                  <h2 className={`text-5xl md:text-6xl font-light tracking-tighter ${
                      currentRisk > 30 ? 'text-red-500' : 
                      currentRisk > 10 ? 'text-amber-500' : 'text-white'
                  }`}>
                      {currentRisk.toFixed(1)}<span className="text-3xl font-light opacity-50">%</span>
                  </h2>
              </div>
              
              <div className="bg-zinc-900/40 border-[0.5px] border-zinc-800/60 rounded-2xl p-6 transition-all hover:bg-zinc-900/60 flex flex-col justify-between">
                  <div className="flex justify-between items-center mb-4">
                    <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Observations</p>
                    <Cpu className="w-4 h-4 text-zinc-600" />
                  </div>
                  <h2 className="text-5xl md:text-6xl font-light text-white tracking-tighter">
                      {data.length ? data[data.length-1].time : 0}
                  </h2>
              </div>
              
               <div className="bg-zinc-900/40 border-[0.5px] border-zinc-800/60 rounded-2xl p-6 transition-all hover:bg-zinc-900/60 flex flex-col justify-between">
                  <div className="flex justify-between items-center mb-4">
                    <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Active Alerts</p>
                    <AlertCircle className={`w-4 h-4 ${alerts.filter(a => a.status === 'pending').length > 0 ? 'text-red-500' : 'text-zinc-600'}`} />
                  </div>
                  <h2 className={`text-5xl md:text-6xl font-light tracking-tighter ${alerts.filter(a => a.status === 'pending').length > 0 ? 'text-red-500' : 'text-white'}`}>
                      {alerts.filter(a => a.status === 'pending').length}
                  </h2>
               </div>
          </div>

          {/* TELEMETRY CHART */}
          <div className="bg-zinc-900/40 border-[0.5px] border-zinc-800/60 rounded-2xl p-6">
            <div className="mb-6">
                <h3 className="text-sm font-medium text-white mb-1">{datasetType === 'custom_model_training' ? 'Training Performance' : 'System Telemetry'}</h3>
                <div className="flex justify-between items-end">
                    <p className="text-xs text-zinc-500">{datasetType === 'custom_model_training' ? `${mlTaskType} Metric Value` : 'System RAM vs Training Steps'}</p>
                    <span className="text-[10px] text-zinc-600 tracking-wider">LIVE</span>
                </div>
            </div>
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} syncId="canarySync" margin={{ top: 5, right: 0, bottom: 5, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                  <XAxis 
                    dataKey="time" 
                    stroke="#52525b" 
                    tick={{fill: '#71717a', fontSize: 11, fontWeight: 300}}
                    tickMargin={12}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis 
                    stroke="#52525b" 
                    tick={{fill: '#71717a', fontSize: 11, fontWeight: 300}}
                    domain={['auto', 'auto']}
                    axisLine={false}
                    tickLine={false}
                    dx={-10}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', color: '#fff', fontSize: '12px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                    itemStyle={{ color: '#e4e4e7' }}
                    cursor={{stroke: '#3f3f46', strokeWidth: 1, strokeDasharray: '4 4'}}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#ffffff" 
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                    activeDot={{ r: 4, fill: '#fff' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* PROBABILITY CHART */}
          <div className="bg-zinc-900/40 border-[0.5px] border-zinc-800/60 rounded-2xl p-6 relative overflow-hidden">
             <div className={`absolute inset-0 opacity-[0.03] transition-opacity duration-1000 pointer-events-none ${
                 currentRisk > 30 ? 'bg-gradient-to-t from-red-500 to-transparent' : 'bg-transparent'
             }`} />
             
            <div className="relative z-10 w-full">
                <div className="mb-6 flex justify-between items-end">
                    <div>
                        <h3 className="text-sm font-medium text-white mb-1">Regime Shift Probability</h3>
                        <p className="text-xs text-zinc-500">Bayesian Change Point Detection Output</p>
                    </div>
                    <div className="text-xs text-red-500 font-medium tracking-wide">
                        THRESHOLD: {(alertThreshold * 100).toFixed(0)}%
                    </div>
                </div>
                <div className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} syncId="canarySync" margin={{ top: 5, right: 0, bottom: 20, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis 
                        dataKey="time" 
                        stroke="#52525b" 
                        tick={{fill: '#71717a', fontSize: 11, fontWeight: 300}}
                        axisLine={false}
                        tickLine={false}
                        tickMargin={12}
                    />
                    <YAxis 
                        stroke="#52525b" 
                        tick={{fill: '#71717a', fontSize: 11, fontWeight: 300}}
                        domain={[0, 1]}
                        ticks={[0, 0.5, 1]}
                        axisLine={false}
                        tickLine={false}
                        dx={-10}
                    />
                    <Tooltip 
                        contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                        itemStyle={{ color: '#ef4444' }}
                        cursor={{stroke: '#3f3f46', strokeWidth: 1, strokeDasharray: '4 4'}}
                    />
                    <ReferenceLine y={alertThreshold} stroke="#ef4444" strokeWidth={1} strokeOpacity={0.6} strokeDasharray="4 4" />
                    <Line 
                        type="stepAfter" 
                        dataKey="probability" 
                        stroke="#ef4444" 
                        strokeWidth={1.5}
                        dot={false}
                        isAnimationActive={false}
                    />
                    <Brush 
                      dataKey="time" 
                      height={20} 
                      stroke="#3f3f46"
                      fill="transparent"
                      tickFormatter={() => ''}
                      travellerWidth={8}
                    />
                    </LineChart>
                </ResponsiveContainer>
                </div>
            </div>
          </div>

        </div>

        {/* SIDEBAR */}
        <div className="xl:col-span-1 flex flex-col h-full space-y-8">
          
          <div className="bg-zinc-900/40 border-[0.5px] border-zinc-800/60 rounded-2xl p-6 flex-1 flex flex-col">
              <h3 className="text-sm font-medium text-white mb-6 tracking-wide flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-zinc-500" />
                Human-in-the-loop
              </h3>
              
              <div className="space-y-4 flex-1 overflow-y-auto pr-2 custom-scrollbar max-h-[500px] xl:max-h-[800px]">
                {alerts.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-center p-8 opacity-40">
                        <CheckCircle2 className="w-6 h-6 mb-3 text-zinc-500" />
                        <p className="text-xs text-zinc-400 font-light">System optimal.<br/>No anomalies detected.</p>
                    </div>
                ) : (
                    alerts.map((alert) => (
                        <div 
                            key={alert.id} 
                            className={`p-4 rounded-xl border-[0.5px] transition-all relative overflow-hidden group
                                ${alert.status === 'pending' ? 'bg-red-950/10 border-red-900/30' : 'bg-transparent border-zinc-800/40 opacity-50 grayscale'}
                            `}
                        >
                            {alert.status === 'pending' && (
                                <div className="absolute left-0 top-0 bottom-0 w-[2px] bg-red-500" />
                            )}
                            
                            <div className="flex justify-between items-start mb-2 pl-1">
                                <div className="flex items-center gap-2">
                                    <span className={`w-1.5 h-1.5 rounded-full ${alert.status === 'pending' ? 'bg-red-500 scale-110' : 'bg-zinc-600'}`} />
                                    <span className="text-xs font-semibold text-white tracking-wide">Shift Detected</span>
                                </div>
                                <span className="text-[10px] text-zinc-500 font-mono">T={alert.time}</span>
                            </div>
                            
                            <p className="text-xs text-zinc-400 mb-4 pl-1 leading-relaxed">
                                Probability: <span className="text-red-400">{alert.risk.toFixed(1)}%</span><br/>
                                Review telemetry data.
                            </p>
                            
                            {alert.status === 'pending' ? (
                                <div className="flex gap-2 pl-1">
                                    <button 
                                        onClick={() => handleAcknowledge(alert.id)}
                                        className="flex-1 py-1.5 bg-zinc-800 text-white hover:bg-zinc-700 rounded-lg text-xs transition-colors"
                                    >
                                        Confirm
                                    </button>
                                    <button 
                                        onClick={() => handleDismiss(alert.id)}
                                        className="flex-1 py-1.5 bg-transparent border border-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-800 rounded-lg text-xs transition-colors"
                                    >
                                        Dismiss
                                    </button>
                                </div>
                            ) : (
                                <div className="text-[10px] uppercase tracking-widest text-zinc-600 pl-1 font-medium mt-1">
                                    {alert.status}
                                </div>
                            )}
                        </div>
                    ))
                )}
              </div>
          </div>

          {(datasetType === 'mobilenet_training' || datasetType === 'custom_model_training' || datasetType === 'csv_upload') && trainingInfo && (
          <div className="bg-zinc-900/40 border-[0.5px] border-zinc-800/60 rounded-2xl p-6">
             <h3 className="text-sm font-medium text-white mb-5 tracking-wide flex items-center gap-2">
                <Sliders className="w-4 h-4 text-zinc-500" />
                Training Configuration
             </h3>
             <div className="space-y-4">
               <div>
                 <p className="text-xs text-zinc-500 mb-1">Hardware Context</p>
                 <div className="flex justify-between items-center text-xs mt-1">
                    <span className="text-zinc-400">RAM/VRAM Detected</span>
                    <span className="text-white font-mono">{trainingInfo.memory_gb} GB</span>
                 </div>
                 <div className="flex justify-between items-center text-xs mt-1">
                    <span className="text-zinc-400">Dataset Size</span>
                    <span className="text-white font-mono">{trainingInfo.dataset_size} imgs</span>
                 </div>
               </div>
               
               <div className="h-px w-full bg-zinc-800/60 my-2" />

               { (datasetType === 'custom_model_training' || datasetType === 'csv_upload') && (
                 <>
                   <div>
                     <div className="flex justify-between items-center mb-2">
                         <label className="text-xs text-zinc-400">Task Type</label>
                     </div>
                     <select 
                         value={mlTaskType}
                         onChange={e => setMlTaskType(e.target.value)}
                         className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-zinc-700 transition-colors"
                     >
                        <option value="Classification">Classification</option>
                        <option value="Regression">Regression</option>
                        <option value="Clustering">Clustering</option>
                     </select>
                   </div>
                   
                   <div>
                     <div className="flex justify-between items-center mb-2 mt-2">
                         <label className="text-xs text-zinc-400">Model Algorithm</label>
                     </div>
                     <select 
                         value={mlModelName}
                         onChange={e => setMlModelName(e.target.value)}
                         className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-zinc-700 transition-colors"
                     >
                        {ML_MODELS[mlTaskType] && ML_MODELS[mlTaskType].map(model => (
                            <option key={model} value={model}>{model}</option>
                        ))}
                     </select>
                   </div>
                   <div className="h-px w-full bg-zinc-800/60 my-3" />
                 </>
               )}

               <div>
                 <div className="flex justify-between items-center mb-2">
                     <label className="text-xs text-zinc-400">Batch Size</label>
                     <span className="text-[10px] text-zinc-500">Predicted: {trainingInfo.predicted_batch_size}</span>
                 </div>
                 <input 
                     type="number"
                     placeholder={trainingInfo.predicted_batch_size}
                     value={customBatchSize}
                     onChange={e => setCustomBatchSize(e.target.value)}
                     className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-zinc-700 transition-colors"
                 />
               </div>

               <div>
                 <div className="flex justify-between items-center mb-2">
                     <label className="text-xs text-zinc-400">Epochs</label>
                     <span className="text-[10px] text-zinc-500">Predicted: {trainingInfo.predicted_epochs}</span>
                 </div>
                 <input 
                     type="number"
                     placeholder={trainingInfo.predicted_epochs}
                     value={customEpochs}
                     onChange={e => setCustomEpochs(e.target.value)}
                     className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-zinc-700 transition-colors"
                 />
               </div>
             </div>
          </div>
          )}

          <div className="bg-zinc-900/40 border-[0.5px] border-zinc-800/60 rounded-2xl p-6">
             <h3 className="text-sm font-medium text-white mb-5 tracking-wide flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-zinc-500" />
                Detection Settings
             </h3>
             <div>
                 <div className="flex justify-between items-center mb-3">
                     <label className="text-xs text-zinc-400">Sensitivity Level</label>
                     <span className="text-xs text-white bg-zinc-800 px-2 py-0.5 rounded-md font-mono">{(alertThreshold * 100).toFixed(0)}%</span>
                 </div>
                 <input 
                     type="range" 
                     min="0.05" 
                     max="0.95" 
                     step="0.05"
                     value={alertThreshold}
                     onChange={(e) => {
                         const val = parseFloat(e.target.value);
                         setAlertThreshold(val);
                         setSettings(val);
                     }}
                     className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-white hover:accent-zinc-300 transition-all"
                 />
             </div>
          </div>

        </div>

      </div>

      {/* RAW DATA TABLE */}
      <div className="mt-8 bg-zinc-900/40 border-[0.5px] border-zinc-800/60 rounded-2xl p-6 overflow-hidden flex flex-col">
        <div className="flex items-center gap-3 mb-6">
           <div className="p-1.5 bg-zinc-900 border border-zinc-800 rounded-lg">
             <Activity className="w-4 h-4 text-zinc-400" />
           </div>
           <div>
              <h3 className="text-sm font-medium text-white">Data Stream</h3>
              <p className="text-[10px] text-zinc-500 uppercase tracking-widest mt-0.5">Latest 50 observations</p>
           </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-zinc-800/60">
                <th className="pb-3 px-4 text-[10px] font-medium text-zinc-500 uppercase tracking-widest">Time</th>
                <th className="pb-3 px-4 text-[10px] font-medium text-zinc-500 uppercase tracking-widest">Value</th>
                <th className="pb-3 px-4 text-[10px] font-medium text-zinc-500 uppercase tracking-widest">Probability</th>
                <th className="pb-3 px-4 text-[10px] font-medium text-zinc-500 uppercase tracking-widest text-right">Status</th>
              </tr>
            </thead>
            <tbody className="text-sm font-light">
              {data.slice().reverse().slice(0, 50).map((row, idx) => {
                const isError = row.probability > alertThreshold;
                return (
                  <tr key={`${row.time}-${idx}`} className="border-b border-zinc-800/30 hover:bg-zinc-800/20 transition-colors group">
                    <td className="py-3 px-4 font-mono text-zinc-400 text-xs">{row.time}</td>
                    <td className="py-3 px-4 font-mono text-white text-xs">{row.value.toFixed(4)}</td>
                    <td className="py-3 px-4 font-mono text-xs">
                      <span className={isError ? 'text-red-400 font-medium' : 'text-zinc-500'}>
                        {(row.probability * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right">
                      {isError ? (
                        <span className="inline-flex items-center gap-1.5 text-xs text-red-500 font-medium tracking-wide">
                          <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></span>
                          Critical
                        </span>
                      ) : (
                        <span className="text-xs text-zinc-600 tracking-wide opacity-0 group-hover:opacity-100 transition-opacity">
                          Normal
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
              {data.length === 0 && (
                <tr>
                  <td colSpan="4" className="py-10 text-center text-zinc-600 text-xs">
                    Waiting for telemetry data...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      
      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar {
            width: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #27272a;
            border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: #3f3f46;
        }
      `}} />
    </div>
  );
};

export default Dashboard;
