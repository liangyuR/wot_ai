#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务管理器模块

持续执行多局战斗循环，处理UI配置的停止条件。
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from loguru import logger
from .battle_task import BattleTask
from src.utils.global_path import GetGlobalConfig

class TaskManager:
    """任务管理器"""
    
    def __init__(
        self,
        run_hours: int = 4,
        auto_stop: bool = False,
        auto_shutdown: bool = False,
        enable_silver_reserve: bool = False,
    ):
        """
        初始化任务管理器
        
        Args:
            run_hours: 运行时长限制（小时）
            auto_stop: 达到时长后自动停止
            auto_shutdown: 达到时长后自动关机（需要管理员权限）
            enable_silver_reserve: 是否启用银币储备功能
        """
        self.run_hours_ = run_hours
        self.auto_stop_ = auto_stop
        self.auto_shutdown_ = auto_shutdown
        self.enable_silver_reserve_ = enable_silver_reserve
        
        self.running_ = False
        self.start_time_ = None
        
        # Load configuration
        try:
            self.config_ = GetGlobalConfig()
            logger.info(f"Task manager loaded config from {self.config_}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise e

        # 定时重启配置
        restart_cfg = self.config_.scheduled_restart
        self.scheduled_restart_enabled_ = restart_cfg.enable
        self.scheduled_restart_hours_ = restart_cfg.interval_hours

    def run_forever(self) -> None:
        """持续执行战斗循环"""
        self.running_ = True
        self.start_time_ = datetime.now()
        
        logger.info(f"任务管理器启动，运行时长限制: {self.run_hours_} 小时")
        if self.scheduled_restart_enabled_:
            logger.info(f"定时重启已启用，间隔: {self.scheduled_restart_hours_} 小时")
        
        try:
            while self.running_:
                # 检查停止条件
                if self._should_stop():
                    logger.info("达到停止条件，退出循环")
                    break

                # 检查定时重启
                if self._should_scheduled_restart():
                    logger.info("达到定时重启时间，准备重启程序...")
                    self._restart_program()
                    return  # execv 会替换进程，不会返回
                
                # 创建战斗任务
                task = BattleTask(enable_silver_reserve=self.enable_silver_reserve_)
                
                # 启动事件驱动循环
                if not task.start():
                    logger.warning("战斗任务启动失败，继续下一局")
                    time.sleep(2.0)
                    continue
                
                # 等待任务运行，直到达到停止条件
                try:
                    while task.is_running() and self.running_:
                        if self._should_stop():
                            logger.info("达到停止条件，停止战斗任务")
                            task.stop()
                            break
                        # 检查定时重启
                        if self._should_scheduled_restart():
                            logger.info("达到定时重启时间，停止当前任务并重启程序...")
                            task.stop()
                            time.sleep(1.0)
                            self._restart_program()
                            return
                        time.sleep(2.0)
                except Exception as e:
                    logger.error(f"战斗任务运行异常: {e}")
                    task.stop()
                
                # 短暂休息
                time.sleep(2.0)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止任务管理器")
        except Exception as e:
            logger.error(f"任务管理器运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running_ = False
            logger.info("任务管理器已停止")
            
            # 如果配置了自动关机
            if self.auto_shutdown_ and self._should_stop():
                self._shutdown()
    
    def stop(self) -> None:
        """停止任务管理器"""
        logger.info("正在停止任务管理器...")
        self.running_ = False
    
    def _should_stop(self) -> bool:
        """
        检查是否应该停止
        
        Returns:
            是否应该停止
        """
        if not self.auto_stop_ and not self.auto_shutdown_:
            return False
        
        if self.start_time_ is None:
            return False
        
        elapsed = datetime.now() - self.start_time_
        elapsed_hours = elapsed.total_seconds() / 3600.0
        
        return elapsed_hours >= self.run_hours_

    def _should_scheduled_restart(self) -> bool:
        """
        检查是否应该执行定时重启
        
        Returns:
            是否应该重启
        """
        if not self.scheduled_restart_enabled_:
            return False
        
        if self.start_time_ is None:
            return False
        
        elapsed = datetime.now() - self.start_time_
        elapsed_hours = elapsed.total_seconds() / 3600.0
        
        return elapsed_hours >= self.scheduled_restart_hours_

    def _restart_program(self) -> None:
        """重启程序（使用 os.execv 替换当前进程）"""
        logger.info("正在重启程序...")

        python = sys.executable
        script = sys.argv[0]
        args = sys.argv[1:]

        # 添加 --auto-start 参数以便重启后自动启动
        if "--auto-start" not in args:
            args.append("--auto-start")

        logger.info(f"重启命令: {python} {script} {' '.join(args)}")

        try:
            os.execv(python, [python, script] + args)
        except Exception as e:
            logger.error(f"重启失败: {e}")
            # 如果 execv 失败，尝试使用 subprocess
            try:
                subprocess.Popen([python, script] + args)
                sys.exit(0)
            except Exception as e2:
                logger.error(f"备用重启也失败: {e2}")
    
    def _shutdown(self) -> None:
        """执行关机操作（需要管理员权限）"""
        logger.info("执行自动关机...")
        
        try:
            subprocess.run(['shutdown', '/s', '/t', '10'], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"关机命令执行失败: {e}")
        except Exception as e:
            logger.error(f"关机操作异常: {e}")

