import os
import time
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from tqdm import tqdm
import imageio

import torch
from torch.utils.data import Dataset, DataLoader


###########################################################
# DATASET CLASS
###########################################################
class GameDataset(Dataset):
    def __init__(self, context_length=64, transform=None):
        self.context_length = context_length
        self.transform = transform

        self.frames = None       # [total_frames, H, W, C]
        self.actions = None      # [total_frames] 
        self.valid_indices = []  # список индексов, где есть контекст и целевой кадр 
        self.current_size = 0
        self.chunk_size = 10000  # чанки для эффективности

    def _initialize_arrays(self, frame_shape, action_dtype=np.int64):
        self.frames = np.zeros((self.chunk_size, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(self.chunk_size, dtype=action_dtype)
        self.current_size = 0

    def _resize_arrays(self, required_size):
        if required_size > len(self.frames):
            new_size = max(required_size, len(self.frames) + self.chunk_size)
            new_frames = np.zeros((new_size, *self.frames.shape[1:]), dtype=np.uint8)
            new_actions = np.zeros(new_size, dtype=self.actions.dtype)
            
            if self.current_size > 0:
                new_frames[:self.current_size] = self.frames[:self.current_size]
                new_actions[:self.current_size] = self.actions[:self.current_size]
            
            self.frames = new_frames
            self.actions = new_actions

    # добавим эпизод
    def add_episode(self, frames, actions):
        if len(frames) <= self.context_length + 1:
            return
            
        if self.frames is None:
            self._initialize_arrays(frames[0].shape)
        
        # проверяем размер
        required_size = self.current_size + len(frames)
        self._resize_arrays(required_size)
        
        # добавляем кадры и действия
        start_idx = self.current_size
        end_idx = start_idx + len(frames)
        
        self.frames[start_idx:end_idx] = frames
        self.actions[start_idx:end_idx] = actions
        
        # добавляем валидные индексы
        for i in range(self.context_length, len(frames) - 1):
            self.valid_indices.append(start_idx + i)
        
        self.current_size = end_idx

    
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        # получаем контекст для кадра
        start = real_idx - self.context_length
        context_frames = self.frames[start:real_idx]    # [64, H, W, C]
        context_actions = self.actions[start:real_idx]  # [64]

        # целевой кадр
        target_frame = self.frames[real_idx+1]  # [H, W, C]

        # в тензоры для обучения
        context_frames = self._frames_to_tensor(context_frames)  # [64, C, H, W]
        context_actions = torch.tensor(context_actions, dtype=torch.long)
        target_frame = self._frames_to_tensor(target_frame)  # [C, H, W]

        return {
            'context_frames': context_frames,
            'context_actions': context_actions,
            'target_frame': target_frame,
        }


    def _frames_to_tensor(self, frames):
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        
        # проверяем, подавался один кадр или целый контекст кадров
        if frames_tensor.ndim == 3:
            frames_tensor = frames_tensor.permute(2, 0, 1)  # HWC -> CHW
        else:
            frames_tensor = frames_tensor.permute(0, 3, 1, 2) # LHWC -> LCHW

        if self.transform:
            frame_tensor = self.transform(frames_tensor)
        return frames_tensor


    def get_space(self):
        if self.frames is None:
            return "no data"
        
        frames_mb = self.frames.nbytes / (1024 * 1024)
        actions_mb = self.actions.nbytes / (1024 * 1024)
        return f'total: {frames_mb + actions_mb} MB'


###########################################################
# GENERATE DATASET
###########################################################
def collect_data(model, env_id, num_episodes=1000, max_steps=1000):
    dataset = GameDataset(context_length=64)
    env = gym.make(env_id, render_mode='rgb_array')

    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        frames = np.zeros((max_steps, 210, 160, 3), dtype=np.uint8)
        actions = np.zeros(max_steps, dtype=np.int64)
        actual_steps = 0

        for step in range(max_steps):
            frame = env.render()
            frames[step] = frame

            # deterministic = False, чтобы было побольше эксплорейшена
            action, _ = model.predict(obs, deterministic=False)
            actions[step] = action

            # двигаемся
            obs, reward, terminated, truncated, info = env.step(action)
            actual_steps = step + 1
            if terminated or truncated:
                break

        # обрезаем до фактического размера (убираем max_steps - actual_steps пустых мест)
        frames = frames[:actual_steps]
        actions = actions[:actual_steps]

        # добвляем эпизод в датасет, если можем из него выделить валидные кадры
        if len(frames) > dataset.context_length + 1:
            dataset.add_episode(frames, actions)


    # финальное обрезание самого датасета
    if dataset.frames is not None:
        dataset.frames = dataset.frames[:dataset.current_size]
        dataset.actions = dataset.actions[:dataset.current_size]

    print(f'датасет собран: {len(dataset)} примеров, {dataset.get_space()}')
    return dataset


###########################################################
# SAVE & LOAD DATASET
###########################################################
def save_dataset(dataset, filepath):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    torch.save({
        'frames': dataset.frames,           
        'actions': dataset.actions,         
        'valid_indices': np.array(dataset.valid_indices),
        'context_length': dataset.context_length,
        'current_size': dataset.current_size
    }, filepath)

def load_dataset(filepath, transform=None):
    data = torch.load(filepath, weights_only=False)
    dataset = GameDataset(context_length=data['context_length'], transform=transform)
    
    dataset.frames = data['frames']
    dataset.actions = data['actions']
    dataset.valid_indices = data['valid_indices'].tolist()
    dataset.current_size = data['current_size']
    
    return dataset


###########################################################
# USAGE
###########################################################
# if __name__ == '__main__':
#     ppo_model = PPO.load("ppo_1m-steps_spaceinvaders")
#     dataset = collect_data(
#         model=ppo_model,
#         env_id='ALE/SpaceInvaders-v5',
#         num_episodes=10,
#         max_steps=1000
#     )

#     print(dataset.get_space())

#     save_dataset(dataset, 'data/test.pt')