@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0peek_dqn.ps1" %*
