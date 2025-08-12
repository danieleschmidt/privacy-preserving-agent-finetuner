{{/*
Expand the name of the chart.
*/}}
{{- define "privacy-finetuner.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "privacy-finetuner.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "privacy-finetuner.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "privacy-finetuner.labels" -}}
helm.sh/chart: {{ include "privacy-finetuner.chart" . }}
{{ include "privacy-finetuner.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "privacy-finetuner.selectorLabels" -}}
app.kubernetes.io/name: {{ include "privacy-finetuner.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Master selector labels
*/}}
{{- define "privacy-finetuner.masterSelectorLabels" -}}
{{ include "privacy-finetuner.selectorLabels" . }}
app.kubernetes.io/component: master
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "privacy-finetuner.workerSelectorLabels" -}}
{{ include "privacy-finetuner.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "privacy-finetuner.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "privacy-finetuner.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the image repository
*/}}
{{- define "privacy-finetuner.image" -}}
{{- if .Values.global.imageRegistry }}
{{- printf "%s/%s" .Values.global.imageRegistry .Values.image.repository }}
{{- else }}
{{- printf "%s/%s" .Values.image.registry .Values.image.repository }}
{{- end }}
{{- end }}

{{/*
Get the master image
*/}}
{{- define "privacy-finetuner.masterImage" -}}
{{- if .Values.global.imageRegistry }}
{{- printf "%s/%s:%s" .Values.global.imageRegistry .Values.master.image.repository .Values.master.image.tag }}
{{- else }}
{{- printf "%s/%s:%s" .Values.image.registry .Values.master.image.repository .Values.master.image.tag }}
{{- end }}
{{- end }}

{{/*
Get the worker image
*/}}
{{- define "privacy-finetuner.workerImage" -}}
{{- if .Values.global.imageRegistry }}
{{- printf "%s/%s:%s" .Values.global.imageRegistry .Values.workers.image.repository .Values.workers.image.tag }}
{{- else }}
{{- printf "%s/%s:%s" .Values.image.registry .Values.workers.image.repository .Values.workers.image.tag }}
{{- end }}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "privacy-finetuner.commonEnv" -}}
- name: PYTHONPATH
  value: "/app"
- name: CUDA_VISIBLE_DEVICES
  value: "0,1"
- name: NCCL_DEBUG
  value: "INFO"
- name: MASTER_ADDR
  value: {{ include "privacy-finetuner.fullname" . }}-master
- name: MASTER_PORT
  value: "29500"
{{- end }}

{{/*
Get storage class
*/}}
{{- define "privacy-finetuner.storageClass" -}}
{{- if .Values.global.storageClass }}
{{- .Values.global.storageClass }}
{{- else }}
{{- .Values.persistence.data.storageClass }}
{{- end }}
{{- end }}

{{/*
Validate configuration
*/}}
{{- define "privacy-finetuner.validateConfig" -}}
{{- if and .Values.autoscaling.hpa.enabled (not .Values.autoscaling.hpa.workers.minReplicas) }}
{{- fail "autoscaling.hpa.workers.minReplicas is required when HPA is enabled" }}
{{- end }}
{{- if and .Values.persistence.data.enabled (not .Values.persistence.data.storageClass) }}
{{- fail "persistence.data.storageClass is required when persistence is enabled" }}
{{- end }}
{{- if and .Values.ingress.enabled (not .Values.ingress.hosts) }}
{{- fail "ingress.hosts is required when ingress is enabled" }}
{{- end }}
{{- end }}

{{/*
Generate certificates
*/}}
{{- define "privacy-finetuner.gen-certs" -}}
{{- $ca := genCA "privacy-finetuner-ca" 3650 }}
{{- $cert := genSignedCert (include "privacy-finetuner.fullname" .) nil (list (printf "%s.%s.svc.cluster.local" (include "privacy-finetuner.fullname" .) .Release.Namespace)) 365 $ca }}
tls.crt: {{ $cert.Cert | b64enc }}
tls.key: {{ $cert.Key | b64enc }}
ca.crt: {{ $ca.Cert | b64enc }}
{{- end }}

{{/*
Resource requirements
*/}}
{{- define "privacy-finetuner.resources" -}}
{{- if .resources }}
resources:
  {{- if .resources.limits }}
  limits:
    {{- range $key, $value := .resources.limits }}
    {{ $key }}: {{ $value | quote }}
    {{- end }}
  {{- end }}
  {{- if .resources.requests }}
  requests:
    {{- range $key, $value := .resources.requests }}
    {{ $key }}: {{ $value | quote }}
    {{- end }}
  {{- end }}
{{- end }}
{{- end }}

{{/*
Node selector
*/}}
{{- define "privacy-finetuner.nodeSelector" -}}
{{- if .nodeSelector }}
nodeSelector:
  {{- range $key, $value := .nodeSelector }}
  {{ $key }}: {{ $value | quote }}
  {{- end }}
{{- end }}
{{- end }}

{{/*
Tolerations
*/}}
{{- define "privacy-finetuner.tolerations" -}}
{{- if .tolerations }}
tolerations:
  {{- toYaml .tolerations | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Affinity
*/}}
{{- define "privacy-finetuner.affinity" -}}
{{- if .affinity }}
affinity:
  {{- toYaml .affinity | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "privacy-finetuner.imagePullSecrets" -}}
{{- if or .Values.global.imagePullSecrets .Values.image.pullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- range .Values.image.pullSecrets }}
  - name: {{ .name }}
{{- end }}
{{- end }}
{{- end }}