import axios from "axios";

const API_URL = "http://localhost:8000/api";

export const getTrainingInfo = async (datasetType = "mobilenet_training") => {
    return axios.get(`${API_URL}/training_info?dataset=${datasetType}`);
}

export const resetSimulation = async (datasetType = "sudden_shift", batchSize = null, epochs = null, mlTaskType = null, mlModelName = null) => {
    const payload = { dataset_type: datasetType };
    if (batchSize !== null) payload.batch_size = parseInt(batchSize, 10);
    if (epochs !== null) payload.epochs = parseInt(epochs, 10);
    if (mlTaskType !== null) payload.ml_task_type = mlTaskType;
    if (mlModelName !== null) payload.ml_model_name = mlModelName;
    return axios.post(`${API_URL}/reset`, payload);
}

export const getStep = async () => {
    return axios.get(`${API_URL}/step`);
}

export const getHistory = async () => {
    return axios.get(`${API_URL}/history`);
}

export const injectAnomaly = async () => {
    return axios.post(`${API_URL}/inject_anomaly`);
}

export const setSettings = async (threshold) => {
    return axios.post(`${API_URL}/settings`, { alert_threshold: threshold });
}

export const stopTraining = async () => {
    return axios.post(`${API_URL}/stop_training`);
}

export const uploadCSV = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    return axios.post(`${API_URL}/upload_csv`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
    });
}
