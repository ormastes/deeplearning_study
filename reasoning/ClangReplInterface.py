from clang_repl_kernel import ClangReplKernel, update_platform_system, is_installed_clang_exist, install_bundles, ClangReplConfig
import platform
import os
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time

class ObjectPool:
    def __init__(self, cls, method, batch_size, *args, **kwargs):
        self.cls = cls                      # class to instantiate
        self.method = method               # unbound method like Worker.process
        self.args = args                   # args for cls(...)
        self.kwargs = kwargs               # kwargs for cls(...)
        self.batch_size = batch_size
        self.pool = []
        self.pool.append(self.cls(*self.args, **self.kwargs))
        self.lock = threading.Lock()
        self.running = True
        self.futures: list[Future] = []
        self.input_map = {}
        self.pool_max = self.batch_size * 2
        self.pool_min = self.batch_size * 2

        self.executor = ThreadPoolExecutor(max_workers=self.pool_max)
        self.creator_executor = ThreadPoolExecutor(max_workers=self.pool_min)
        self.thread = threading.Thread(target=self._maintain_pool, daemon=True)
        self.thread.start()
        while len(self.pool) < self.pool_min:
            time.sleep(0.05)

    def _maintain_pool(self):
        while self.running:
            with self.lock:
                need = self.pool_max - len(self.pool)
    
            if need > 0:
                futures = []
                for _ in  range(need):
                    futures.append(self.creator_executor.submit(self.cls, *self.args, **self.kwargs))
                    time.sleep(0.1)
                
                for f in futures:
                    obj = f.result()
                    if obj.ok:
                        with self.lock:
                            self.pool.append(obj)
    
            time.sleep(0.1)

    def get_objects(self, n):
        while True:
            with self.lock:
                if len(self.pool) >= n:
                    return [self.pool.pop(0) for _ in range(n)]
            time.sleep(0.05)

    def start_tasks(self, args_list):
        objs = self.get_objects(len(args_list))
        time.sleep(0.1)
        futures = []
        for obj, args in zip(objs, args_list):
            f = self.executor.submit(self.method, obj, *args)
            self.input_map[f] = (obj, args)
            futures.append(f)
            time.sleep(0.1)
        self.futures.extend(futures)
        time.sleep(0.1)

    def log_timeout(self, obj, args, log_header="", log_post=""):
        a_min_load, _, _ = os.getloadavg()
        progress = obj.get_progress()
        score = 1.24 + progress * 0.8 if progress > 0.0 else 0
        
        error_msg = (f"{log_header}TimeoutError (ObjectPool.get_results): "
                f"score={score:.2f}, CPU load={a_min_load:.2f}/32{log_post}")
        print(error_msg)
        return (score, error_msg)

    def get_results(self, timeout=60.0):
        collected = []
        done_futures = []
        while len(self.pool) < self.pool_min:
            time.sleep(0.05)
    
        for f in self.futures:
            try:
                result = f.result(timeout=timeout)
            except TimeoutError:
                # First timeout: try once more
                obj, args = self.input_map[f]
                result = self.log_timeout(obj, args, log_post=", input:" + str(args))

                # Retry once: re-submit the same task.
                new_obj = self.get_objects(1)[0]
                retry_future = self.executor.submit(self.method, new_obj, *args)
                try:
                    result = retry_future.result(timeout=timeout*2)
                    print("Retry succeeded after retry")
                except TimeoutError:
                    result = self.log_timeout(new_obj, args, log_header="Retry failed: ")
                
            except Exception as e:
                result =(0.0, f"Exeception {e}")
                print("Exception:", e)
            finally:
                if not isinstance(result, (list, tuple)):
                    result = [result]
                collected.append(result)
                done_futures.append(f)

        # Remove processed futures
        for f in done_futures:
            if f in self.futures:
                del self.input_map[f]
                self.futures.remove(f)
    
        # Transpose
        transposed = list(map(list, zip(*collected)))

        return transposed if len(transposed) > 1 else transposed[0]

    def stop(self):
        self.running = False
        self.thread.join()
        self.executor.shutdown(wait=True)


class ClangReplInterface:
    def __init__(self):
        if not is_installed_clang_exist():
            install_bundles(platform.system())
        if not ClangReplKernel.ClangReplKernel_InTest:
            ClangReplKernel.ClangReplKernel_InTest = True
        self.line_count = sys.maxsize
        self.outputs = []
        try:
            self.kernel = None
            self.kernel = ClangReplKernel() 
            self.shell = None
            self.shell = self.kernel.my_shell
            self.kernel.my_shell.run()
            state, log = self.run_verify(">>> %<<0\n0")
            self.execution_count = 0
            self.ok = True
        except Exception as e:
            try:
                with open("runs/clange_repl_dump.pkl", "wb") as f:
                    pickle.dump(self, f)
                print("ClangReplInterface creation error:", e)
    
                if getattr(self, "kernel", None) is None:
                    print("self.kernel is None")
                elif getattr(self.kernel, "my_shell", None) is None:
                    print("self.kernel.my_shell is None")
                else:
                    if getattr(self.kernel.my_shell, "program_with_args", None) is None:
                        print("self.kernel.my_shell.program_with_args is None")
                    else:
                        print("ClangReplInterface command line:", self.kernel.my_shell.program_with_args)
                    if getattr(self.kernel.my_shell, "args", None) is None:
                        print("self.kernel.my_shell.args is None")
                    else:
                        print("ClangReplInterface args:", self.kernel.my_shell.args)
                    if getattr(self.kernel.my_shell, "env", None) is None:
                        print("self.kernel.my_shell.env is None")
                    else:
                        print("ClangReplInterface env:", self.kernel.my_shell.env)
            except Exception as e2:
                print("ClangReplInterface creation error in exception block:", e2)

            self.ok = False

    def get_progress(self):
        return len(self.outputs)/self.line_count

    def do_execute_sync(self, code):
        return self.kernel.do_execute_sync(code)

    def run_verify(self, codes):
        lines = codes.splitlines()
        self.line_count = len(lines)
        for idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith(">>>"):
                line = line[3:].strip()
                if len(line) == 0:
                    self.outputs.append('')
                    continue
                result = self.do_execute_sync(line)
                if result['status'] != 'ok':
                    return result['status'], "Output lists:"+self.outputs
                self.outputs.append(result['output'].strip())
            else:
                if line == 'true' and self.outputs[-1] == '1':
                    pass
                elif line == 'false' and self.outputs[-1] == '0':
                    pass
                elif self.outputs[-1] != line :
                    return "fail", "Expected Output: "+line+", Actual Output: "+ self.outputs[-1]
                self.outputs.append('')
        if self.outputs[-1] == '':
            return 'ok', ''
        else:
            return "fail", "Expected Output at last: '', Output lists:"+self.outputs
                    
                
class ClangReplInterfacePool(ObjectPool):     
    def __init__(self, batch_size):
        super().__init__(self, ClangReplInterface, ClangReplInterface.run_verify, batch_size)

    def run_verify(self, codes):
        self.start_tasks(codes)
        
    def take_result(self):
        return self.get_results()
        
