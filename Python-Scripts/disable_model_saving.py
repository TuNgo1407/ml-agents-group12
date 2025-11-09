import os
import site
import shutil

def disable_model_saving():
    mlagents_paths = []
    for path in site.getsitepackages():
        mlagents_path = os.path.join(path, 'mlagents')
        if os.path.exists(mlagents_path):
            mlagents_paths.append(mlagents_path)
    
    if not mlagents_paths:
        raise RuntimeError("Could not find mlagents installation")

    mlagents_path = mlagents_paths[0]
    saver_path = os.path.join(mlagents_path, 'trainers/model_saver/torch_model_saver.py')
    if os.path.exists(saver_path):
        with open(saver_path, 'r') as f:
            content = f.read()
        
        content = content.replace(
            'def save_checkpoint(self, behavior_name: str, step: int) -> Tuple[str, List[str]]:\n',
            'def save_checkpoint(self, behavior_name: str, step: int) -> Tuple[str, List[str]]:\n        return "", []\n'
        )
        
        content = content.replace(
            'def export(self, output_filepath: str, behavior_name: str) -> None:\n',
            'def export(self, output_filepath: str, behavior_name: str) -> None:\n        pass\n'
        )
        
        with open(saver_path, 'w') as f:
            f.write(content)

    serialization_path = os.path.join(mlagents_path, 'trainers/torch_entities/model_serialization.py')
    if os.path.exists(serialization_path):
        with open(serialization_path, 'r') as f:
            content = f.read()
        
        content = content.replace(
            'def export_policy_model(self, output_filepath: str) -> None:\n',
            'def export_policy_model(self, output_filepath: str) -> None:\n        pass\n'
        )
        
        with open(serialization_path, 'w') as f:
            f.write(content)

    trainer_path = os.path.join(mlagents_path, 'trainers/trainer/rl_trainer.py')
    if os.path.exists(trainer_path):
        with open(trainer_path, 'r') as f:
            content = f.read()
        
        content = content.replace(
            'def save_model(self) -> None:\n',
            'def save_model(self) -> None:\n        pass\n'
        )
        
        content = content.replace(
            'def _maybe_save_model(self, step_after_process: int) -> None:\n',
            'def _maybe_save_model(self, step_after_process: int) -> None:\n        pass\n'
        )
        
        with open(trainer_path, 'w') as f:
            f.write(content)

    print("Successfully disabled model saving and ONNX export in ML-Agents")

if __name__ == '__main__':
    disable_model_saving()
