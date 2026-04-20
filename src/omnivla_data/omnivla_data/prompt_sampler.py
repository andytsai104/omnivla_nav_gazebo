#!/usr/bin/env python3
"""
prompt_sampler.py
-----------------
Generates diverse natural language navigation prompts for a given goal.
Supports three modes:
  - 'label': Uses the exact display label.
  - 'aliases': Randomly picks an alias from the goal definition.
  - 'template': Combines labels/aliases with templates loaded from a YAML file.
"""

import yaml
import random
from pathlib import Path

class PromptSampler:
    def __init__(self, mode: str = 'template', template_yaml_path: str = None):
        """
        :param mode: 'label', 'aliases', or 'template'
        :param template_yaml_path: Path to the YAML file containing 'templates' list
        """
        self.mode = mode.lower()
        self.templates = []
        
        # Load from YAML if provided, otherwise fallback to defaults
        if template_yaml_path and Path(template_yaml_path).exists():
            self._load_templates(template_yaml_path)
        else:
            self._load_defaults()

    def _load_templates(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.templates = data.get('templates', [])
            
        if not self.templates:
            print(f"[PromptSampler] Warning: No templates found in {path}. Using defaults.")
            self._load_defaults()

    def _load_defaults(self):
        """Fallback templates if YAML is missing or empty."""
        self.templates = [
            "Go to the {target}.",
            "Navigate to the {target}."
        ]

    def sample(self, goal: dict) -> str:
        """
        Samples a prompt based on the configured mode and goal dictionary.
        """
        fallback = goal.get('id', 'unknown_goal')
        
        if self.mode == 'label':
            return goal.get('label', fallback)
            
        elif self.mode == 'aliases':
            aliases = goal.get('aliases', [])
            if aliases:
                return random.choice(aliases)
            return goal.get('label', fallback)
            
        elif self.mode == 'template':
            targets = goal.get('aliases', [])
            if goal.get('label'):
                targets.append(goal['label'])
                
            if not targets:
                targets = [fallback]
                
            target = random.choice(targets)
            template = random.choice(self.templates)
            
            prompt = template.format(target=target)
            return prompt[:1].upper() + prompt[1:]
            
        else:
            return fallback