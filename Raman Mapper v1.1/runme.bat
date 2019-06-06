echo off
echo                                                                         ::
echo                   Add necessary directories to environment path

IF EXIST %USERPROFILE%\Anaconda3 SET PATH=%PATH%;%USERPROFILE%\Anaconda3;%USERPROFILE%\Anaconda3\Scripts;%USERPROFILE%\Anaconda3\Library\mingw-w64\bin;%USERPROFILE%\Anaconda3\Library\bin
echo Done
echo _______

echo                                                                         ::
echo                   Start BOKEH server with python connection
echo _______

echo                                                                         ::
bokeh serve  --show bokeh_app.py
