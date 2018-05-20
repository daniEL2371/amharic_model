$global:condaenv = '';
$global:e = 0;

function prompt{

  $p = Split-Path -leaf -path (Get-Location)

  if($e.Equals(0)){

      echo "$p> ";

  }elseif($e.Equals(1)){

      echo "(CPU-TF) $p> ";

  }
  elseif($e.Equals(2)){

      echo "(GPU-TF) $p> ";

  }

}

# Modify to your actual anaconda path
$CondaPath = "C:\ProgramData\Anaconda3"

function Register-Conda($EnvName) {
   
    $envPath = "C:\Users\amany\AppData\Local\conda\conda\envs\$EnvName"
    
    $env:Path = $($env:Path -Split ";" | ? { !($_.StartsWith($CondaPath)) }) -Join ";"
    $env:Path = "$envPath\Scripts" + ";" + $env:Path
    $env:Path = $envPath + ";" + $env:Path
    
    $env:CONDA_DEFAULT_ENV = $envPath
    $env:CONDA_PREFIX = $envPath
};


function cpu(){
    $condaenv = 'cputf3';
    $global:e = 1;
    Register-Conda $condaenv;
}

function gpu(){
    $condaenv = 'gputf3';
    $global:e = 2
    Register-Conda $condaenv;
}


Set-Alias cputf cpu;
Set-Alias gputf gpu;