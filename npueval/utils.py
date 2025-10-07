# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from CppHeaderParser import CppHeader
import numpy as np
import subprocess
import json
import re
import os
from pathlib import Path

def get_kernel_code(test: dict, solutions_path: str = None) -> str:
    """Fetch the kernel code from the provided solution path, if none provided default 
    to canonical solution."""
    if not solutions_path:
        return test['prompt'] + test['canonical_solution']
    
    with open(os.path.join(solutions_path, f"{test['kernel_name']}.json"), 'r') as sol_file:
        solution = json.load(sol_file)
        if not solution.get('code'):
            print(f"No code available in {solutions_path} for {test['kernel_name']}")
            return None

        srccode = solution['code']
        
        # if gpt decides to be too helpful and adds a main()... remove it
        srccode = re.sub(r'int\s+main\s*\([^)]*\)\s*{[^{}]*({[^{}]*}[^{}]*)*}', '', srccode, flags=re.DOTALL)
        
        # cppheaderparser will complain if we don't remove trailing comments
        srccode = srccode.split('// extern "C"')[0]
        
        return srccode

def extract_buffers(test):
    """Specific helper for the AIEval dataset - parses the test dictionary and returns
    input buffers, output buffers and RTPs as separate lists.
    """
    input_buffers = []
    for x in test['test_vectors']['inputs']:
        array, dtype = list(x.values())
        input_buffers.append(np.array(array, dtype=dtype))
    
    output_buffers = []
    for x in test['test_vectors']['outputs']:
        array, dtype = list(x.values())
        output_buffers.append(np.array(array, dtype=dtype))

    rtps = []
    if test['test_vectors'].get('rtps') != None:
        for rtp in test['test_vectors']['rtps']:
            array, dtype = rtp.values()
            rtps.append(np.array(array, dtype=dtype))
            # rtp_names.append(list(rtp.keys())[0])
    
    return input_buffers, output_buffers, rtps

def trace_to_json(trace_file: str, mlir_file: str, output_name: str="trace.json", dev="npu1"):
    """Subprocesses wrapper over parse_trace.py utility.

    Parameters
    ----------
    trace_file : str
        The .txt trace file of 32-byte codes.
    mlir_file : str
        Path to the corresponding MLIR file for the design being traced.
    output_name : str, optional
        Path to output json file. You can analyze it using tools like https://ui.perfetto.dev
    """
    
    if dev == "npu1":
        colshift = "1"
    elif dev == "npu2":
        colshift = "0"
    else:
        raise Exception("Unsupported device")
    
    command = [
        os.environ["MLIR_AIE_BUILD_DIR"]+"/programming_examples/utils/parse_trace.py",
        "--input",
        trace_file,
        "--mlir",
        mlir_file,
        "--colshift",
        colshift,
        "--output",
        output_name
    ]

    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        if result.returncode == 0:
            print(f"Trace written to {output_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Trace failed\n{e.output}")
        return e.output

def get_cycles(trace_path):
    """This helper function should only be used to extract cycle counts
    from NPUEval trace files where the expectation is to have exactly 1 of
    each event0 and event1.
    """
    with open(trace_path, 'r') as f:
        data = json.load(f)

    event0 = []
    event1 = []
    try:
        for x in data:
            if (x['name'] == "INSTR_EVENT_0") and (x['ph'] == 'B'):
                event0.append(x['ts'])
        
            if x['name'] == "INSTR_EVENT_1" and x['ph'] == 'B':
                event1.append(x['ts'])
        
        return event1[0]-event0[0]
    except:
        return np.inf

def get_vector_time(trace, return_score=True):
    """This function extracts the total time spent on the vectorized unit
    from an NPUEval AIE trace (this must have exactly 1 event0 and 1 event1
    sandwiching the kernel call).
    """
    with open(trace, 'r') as f:
        data = json.load(f)

    start, end = None, None
    
    # find start and end
    for x in data:
        if not start:
            if (x['name'] == "INSTR_EVENT_0") and (x['ph'] == 'B'):
                start = x['ts']
        if x['name'] == "INSTR_EVENT_1" and x['ph'] == 'B':
            end = x['ts']

    if not start or not end:
        return 0

    total_duration = 0
    stack = []
    
    for event in data:
        if event['name'] == "INSTR_VECTOR":
            if event['ts'] < start:
                continue
            
            if event['ts'] > end:
                continue
            
            if event['ph'] == 'B':
                stack.append(event)
            elif event['ph'] == 'E' and stack:
                # Get matching begin event
                begin_event = stack.pop()
                # Calculate duration for this pair
                duration = event['ts'] - begin_event['ts']
                total_duration += duration
            
    if return_score:
        return total_duration/(end-start)
    else:
        return total_duration

def parse_stack_sizes(asm_path):
    with open(asm_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    current_function = None

    for i, line in enumerate(lines):
        # Match the function name from `.size` directive
        size_match = re.match(r'\s*\.size\s+([^\s,]+)', line)
        if size_match:
            current_function = size_match.group(1)

        # Match the .section .stack_sizes marker
        if '.section\t.stack_sizes' in line or '.section .stack_sizes' in line:
            # Get the next 1-2 lines which should contain stack size
            if i + 2 < len(lines):
                stack_lines = lines[i+1:i+3]
                for sl in stack_lines:
                    byte_match = re.search(r'\.byte\s+(\d+)', sl)
                    ascii_match = re.search(r'\.ascii\s+"((?:\\\d{3})+)"', sl)
                    if byte_match:
                        size = int(byte_match.group(1))
                        results.append((current_function, size))
                        break
                    elif ascii_match:
                        ascii_bytes = bytes(int(b, 8) for b in re.findall(r'\\(\d{3})', ascii_match.group(1)))
                        size = int.from_bytes(ascii_bytes, byteorder='little')
                        results.append((current_function, size))
                        break
    return results

def report_peano_version():
    """Returns parsed Peano version information as a dictionary.
    """
    # Check if peano dir is set
    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    if not peano_dir:
        raise EnvironmentError("PEANO_INSTALL_DIR environment variable is not set.")
    
    command = [f"{peano_dir}/bin/clang", "--version"]
    result = subprocess.run(command, capture_output=True, check=True, text=True)
    output = result.stdout.strip()

    # Regex parse the version string
    version_info = {}
    if m := re.search(r'clang version ([0-9.]+) \((https://[^ ]+) ([a-f0-9]+)\)', output):
        version_info["version"] = m.group(1)
        version_info["repo_url"] = m.group(2)
        version_info["commit_hash"] = m.group(3)
    
    # Extract build config flags
    if m := re.search(r'Build config: ([^\n]+)', output):
        version_info["build_config"] = m.group(1).split()
    
    return version_info

def report_xdna_version(report_path="report.json", printout=False):
    """Returns device info and xdna version.
    """
    command = ["xrt-smi", "examine", "-f", "JSON", "-o", report_path, "--force"]
    
    result = subprocess.run(command, capture_output=True, check=True, text=True)

    if printout:
        print(result.stdout.strip())

    with open(report_path) as f:
        data = json.load(f)

    compiler_version = report_peano_version()
    
    system_info = {'device': data['system']['host']['devices'][0]['name'],
               'os': data['system']['host']['os']['distribution'],
               'kernel': data['system']['host']['os']['release'],
               'xdna_version': data['system']['host']['xrt']['drivers'][0]['version'],
               'xdna_hash': data['system']['host']['xrt']['drivers'][0]['hash'],
               'firmware_version': data['system']['host']['devices'][0]['firmware_version'],
               'compiler_version': compiler_version}

    return system_info
