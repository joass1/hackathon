"""
Enhanced 3D Vision System for Robotic Arm Manipulation
========================================================
Specifically addresses hackathon requirements:
✓ Processing 3D sensor/camera data (depth maps, point clouds)
✓ Object detection & segmentation with occlusion handling
✓ 6-DOF pose estimation for stacked objects
✓ Grasp planning for partially occluded objects
"""

import cv2
import numpy as np
import time
import logging
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except:
    torch = None
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("✅ YOLO available for object detection")
except:
    YOLO_AVAILABLE = False
    logger.warning("⚠️ YOLO not available. Install with: pip install ultralytics")

# =============================================================================
# CAMERA INTRINSICS
# =============================================================================

@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_fov(width: int, height: int, fov_degrees: float = 60.0) -> "CameraIntrinsics":
        """Approximate focal length from horizontal FOV"""
        fov = np.deg2rad(fov_degrees)
        fx = (width / 2.0) / np.tan(fov / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return CameraIntrinsics(width, height, fx, fy, cx, cy)

# =============================================================================
# DEPTH ESTIMATION WITH OCCLUSION AWARENESS
# =============================================================================

class OcclusionAwareDepthEstimator:
    """
    Enhanced depth estimation that handles occlusions and discontinuities
    """
    
    def __init__(self, camera_intrinsics: CameraIntrinsics, reference_width_m: float = 0.3):
        self.intrinsics = camera_intrinsics
        self.reference_width_m = reference_width_m
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.midas = None
        self.transform = None
        
        if TORCH_AVAILABLE:
            try:
                logger.info("Loading MiDaS for depth estimation...")
                self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                self.midas.to(self.device).eval()
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self.transform = transforms.dpt_transform
                logger.info("✅ MiDaS loaded successfully")
            except Exception as e:
                logger.warning(f"MiDaS unavailable: {e}. Using fallback depth.")
                self.midas = None

    def estimate_depth_with_occlusion_map(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns depth map (meters) and occlusion confidence map (0-1)
        Occlusion map indicates depth discontinuities where objects overlap
        """
        h, w = image.shape[:2]
        
        # Get raw depth
        if self.midas is not None and self.transform is not None:
            depth_rel = self._midas_depth(image)
        else:
            depth_rel = self._heuristic_depth(image)
        
        # Scale to meters
        depth_m = self._scale_to_meters(depth_rel)
        
        # Compute occlusion map from depth discontinuities
        occlusion_map = self._compute_occlusion_map(depth_m)
        
        return depth_m, occlusion_map
    
    def _midas_depth(self, image: np.ndarray) -> np.ndarray:
        """MiDaS depth estimation"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        return depth.cpu().numpy().astype(np.float32)
    
    def _heuristic_depth(self, image: np.ndarray) -> np.ndarray:
        """Fast fallback depth using edges and gradients"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Vertical gradient base
        depth = np.linspace(1.0, 5.0, h).reshape(h, 1)
        depth = np.repeat(depth, w, axis=1)
        
        # Edge-based refinement
        edges = cv2.Canny(gray, 50, 150)
        depth[edges > 0] *= 0.8
        
        # Blur variation
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_map = cv2.GaussianBlur(np.abs(laplacian), (21, 21), 0)
        blur_norm = blur_map / (blur_map.max() + 1e-6)
        depth = depth * (1.0 - 0.3 * blur_norm)
        
        return cv2.GaussianBlur(depth.astype(np.float32), (15, 15), 0)
    
    def _scale_to_meters(self, depth_rel: np.ndarray) -> np.ndarray:
        """Scale relative depth to metric depth"""
        # Simple global scaling based on scene statistics
        median_depth = np.median(depth_rel)
        if median_depth > 0:
            scale = 2.0 / median_depth  # Assume median depth is ~2 meters
        else:
            scale = 1.0
        
        depth_m = depth_rel * scale
        depth_m = np.clip(depth_m, 0.1, 10.0)
        return depth_m
    
    def _compute_occlusion_map(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Detect occlusion boundaries from depth discontinuities
        Returns confidence map where 1.0 = likely occlusion boundary
        """
        # Compute depth gradients
        grad_x = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude
        grad_norm = grad_mag / (np.percentile(grad_mag, 95) + 1e-6)
        grad_norm = np.clip(grad_norm, 0, 1)
        
        # Dilate to create occlusion regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occlusion_map = cv2.dilate(grad_norm, kernel, iterations=2)
        
        return occlusion_map

# =============================================================================
# STACKING DETECTION AND ANALYSIS
# =============================================================================

class StackingAnalyzer:
    """
    Detects and analyzes stacked objects using depth and segmentation
    """
    
    def __init__(self, z_threshold_m: float = 0.15):
        self.z_threshold = z_threshold_m  # Max vertical distance for stacking
    
    def detect_stacks(self, detections: List[Dict], depth_map: np.ndarray) -> Dict:
        """
        Analyze stacking relationships between detected objects
        Returns stacking graph with relationships and stability scores
        """
        stacks = []
        relationships = []
        
        # Extract 3D centroids
        objects_3d = []
        for det in detections:
            if det.get('centroid_camera') is not None:
                objects_3d.append({
                    'id': len(objects_3d),
                    'detection': det,
                    'centroid': np.array(det['centroid_camera']),
                    'bbox': det['bbox']
                })
        
        # Find vertical stacking relationships
        for i, obj1 in enumerate(objects_3d):
            for j, obj2 in enumerate(objects_3d[i+1:], i+1):
                rel = self._check_stacking(obj1, obj2)
                if rel:
                    relationships.append(rel)
        
        # Group into stacks
        stack_groups = self._group_stacks(objects_3d, relationships)
        
        # Analyze stack stability
        for stack in stack_groups:
            stability = self._compute_stack_stability(stack, objects_3d)
            stacks.append({
                'objects': stack,
                'height': len(stack),
                'stability': stability,
                'base_object': stack[0] if stack else None,
                'top_object': stack[-1] if stack else None
            })
        
        return {
            'stacks': stacks,
            'relationships': relationships,
            'num_stacked': sum(len(s['objects']) for s in stacks),
            'max_stack_height': max([s['height'] for s in stacks]) if stacks else 0
        }
    
    def _check_stacking(self, obj1: Dict, obj2: Dict) -> Optional[Dict]:
        """Check if obj2 is stacked on obj1"""
        c1, c2 = obj1['centroid'], obj2['centroid']
        
        # Check vertical alignment
        dx = abs(c2[0] - c1[0])
        dy = abs(c2[1] - c1[1])
        dz = c2[2] - c1[2]
        
        # Compute bbox overlap
        overlap = self._bbox_overlap_ratio(obj1['bbox'], obj2['bbox'])
        
        # Stacking criteria:
        # 1. Vertical separation within threshold
        # 2. Horizontal alignment
        # 3. Sufficient bbox overlap
        if (0 < dz < self.z_threshold and 
            dx < 0.05 and dy < 0.05 and 
            overlap > 0.3):
            
            return {
                'type': 'stacked_on',
                'bottom': obj1['id'],
                'top': obj2['id'],
                'z_gap': dz,
                'overlap': overlap,
                'confidence': min(1.0, overlap * (1.0 - dz/self.z_threshold))
            }
        
        return None
    
    def _bbox_overlap_ratio(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute IoU-like overlap ratio for 2D bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return inter_area / min(area1, area2) if min(area1, area2) > 0 else 0
    
    def _group_stacks(self, objects: List[Dict], relationships: List[Dict]) -> List[List[int]]:
        """Group objects into vertical stacks"""
        # Build adjacency for stacking
        stacking_graph = {obj['id']: [] for obj in objects}
        
        for rel in relationships:
            if rel['type'] == 'stacked_on':
                stacking_graph[rel['bottom']].append(rel['top'])
        
        # Find connected components (stacks)
        visited = set()
        stacks = []
        
        for obj_id in stacking_graph:
            if obj_id not in visited:
                stack = []
                self._dfs_stack(obj_id, stacking_graph, visited, stack)
                if len(stack) > 1:
                    # Sort by Z coordinate (bottom to top)
                    stack.sort(key=lambda x: objects[x]['centroid'][2])
                    stacks.append(stack)
        
        return stacks
    
    def _dfs_stack(self, node: int, graph: Dict, visited: set, stack: List):
        """DFS to find connected stack components"""
        visited.add(node)
        stack.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                self._dfs_stack(neighbor, graph, visited, stack)
    
    def _compute_stack_stability(self, stack: List[int], objects: List[Dict]) -> float:
        """Estimate stability of a stack (0-1, higher is more stable)"""
        if len(stack) <= 1:
            return 1.0
        
        stability = 1.0
        
        # Check center of mass alignment
        for i in range(len(stack) - 1):
            lower = objects[stack[i]]['centroid']
            upper = objects[stack[i+1]]['centroid']
            
            # Penalize horizontal offset
            offset = np.sqrt((upper[0] - lower[0])**2 + (upper[1] - lower[1])**2)
            stability *= max(0.3, 1.0 - offset * 5)
        
        # Penalize tall stacks
        height_penalty = max(0.5, 1.0 - len(stack) * 0.15)
        stability *= height_penalty
        
        return max(0.0, min(1.0, stability))

# =============================================================================
# ENHANCED OBJECT DETECTOR WITH OCCLUSION HANDLING
# =============================================================================

class OcclusionAwareDetector:
    """
    Object detection with occlusion reasoning and instance segmentation
    """
    
    def __init__(self, model_size='n', confidence_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = {}
        
        if YOLO_AVAILABLE:
            try:
                # Try segmentation model first for better occlusion handling
                model_name = f'yolov8{model_size}-seg.pt'
                logger.info(f"Loading YOLO segmentation model: {model_name}")
                self.model = YOLO(model_name)
                self.class_names = self.model.names
                logger.info(f"✅ Model loaded with {len(self.class_names)} classes")
            except:
                # Fallback to detection model
                try:
                    model_name = f'yolov8{model_size}.pt'
                    self.model = YOLO(model_name)
                    self.class_names = self.model.names
                except:
                    logger.error("Failed to load YOLO")
                    self.model = None
        
        # Color palette
        self.colors = self._generate_colors(80)
        
    def _generate_colors(self, n):
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), 
                                cv2.COLOR_HSV2BGR)[0, 0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def detect_with_occlusion_handling(self, image: np.ndarray, 
                                      depth_map: np.ndarray, 
                                      occlusion_map: np.ndarray) -> List[Dict]:
        """
        Detect objects and handle occlusions using depth and occlusion maps
        """
        if self.model is None:
            return []
        
        h, w = image.shape[:2]
        detections = []
        
        # Run YOLO
        results = self.model(image, verbose=False, conf=self.confidence_threshold)
        
        for result in results:
            boxes = result.boxes if hasattr(result, 'boxes') else None
            masks = result.masks if hasattr(result, 'masks') else None
            
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                detection = self._process_detection(i, boxes, masks, image, 
                                                   depth_map, occlusion_map)
                if detection:
                    detections.append(detection)
        
        # Analyze occlusions between detections
        self._analyze_mutual_occlusions(detections, depth_map)
        
        return detections
    
    def _process_detection(self, idx: int, boxes, masks, image: np.ndarray,
                          depth_map: np.ndarray, occlusion_map: np.ndarray) -> Dict:
        """Process single detection with occlusion analysis"""
        
        # Extract bbox
        x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get mask if available
        mask = None
        if masks is not None:
            try:
                mask_data = masks.data if hasattr(masks, 'data') else masks
                mask = mask_data[idx].cpu().numpy()
                mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]))
                mask = mask > 0.5
            except:
                mask = None
        
        # If no mask, create from bbox
        if mask is None:
            mask = np.zeros(depth_map.shape, dtype=bool)
            mask[y1:y2, x1:x2] = True
        
        # Compute occlusion score for this object
        occlusion_roi = occlusion_map[y1:y2, x1:x2]
        occlusion_score = float(np.mean(occlusion_roi)) if occlusion_roi.size > 0 else 0
        
        # Estimate visible percentage
        if mask is not None:
            total_pixels = mask.sum()
            occluded_pixels = (mask & (occlusion_map > 0.5)).sum()
            visible_ratio = 1.0 - (occluded_pixels / max(1, total_pixels))
        else:
            visible_ratio = 1.0 - occlusion_score
        
        detection = {
            'bbox': (x1, y1, x2, y2),
            'confidence': float(boxes.conf[idx]),
            'class_id': int(boxes.cls[idx]),
            'label': self.class_names.get(int(boxes.cls[idx]), 'unknown'),
            'mask': mask,
            'occlusion_score': occlusion_score,
            'visible_ratio': visible_ratio,
            'mean_depth': float(np.mean(depth_map[mask])) if mask is not None else 0,
            'depth_std': float(np.std(depth_map[mask])) if mask is not None else 0
        }
        
        return detection
    
    def _analyze_mutual_occlusions(self, detections: List[Dict], depth_map: np.ndarray):
        """Analyze which objects occlude others"""
        
        # First, initialize occlusion lists for ALL detections
        for det in detections:
            det['occluding'] = []
            det['occluded_by'] = []
        
        # Then analyze relationships between all pairs
        for i in range(len(detections)):
            for j in range(len(detections)):
                if i == j:
                    continue
                
                det1 = detections[i]
                det2 = detections[j]
                
                # Check if bboxes overlap
                overlap = self._compute_overlap(det1['bbox'], det2['bbox'])
                
                if overlap > 0:
                    # Compare depths - det1 occludes det2 if det1 is closer
                    if det1.get('mean_depth', 0) < det2.get('mean_depth', 0) - 0.1:
                        if j not in det1['occluding']:
                            det1['occluding'].append(j)
                        if i not in det2['occluded_by']:
                            det2['occluded_by'].append(i)

    def _compute_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute overlap area between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0
        
        return (x2_i - x1_i) * (y2_i - y1_i)

# =============================================================================
# POINT CLOUD PROCESSING
# =============================================================================

def generate_point_cloud_from_depth(depth_map: np.ndarray, 
                                   intrinsics: CameraIntrinsics,
                                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Generate 3D point cloud from depth map"""
    h, w = depth_map.shape
    
    # Create pixel grid
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply mask if provided
    if mask is not None:
        valid = mask
    else:
        valid = np.ones((h, w), dtype=bool)
    
    # Back-project to 3D
    z = depth_map[valid]
    x = (xx[valid] - intrinsics.cx) * z / intrinsics.fx
    y = (yy[valid] - intrinsics.cy) * z / intrinsics.fy
    
    # Filter invalid points
    valid_pts = (z > 0.1) & (z < 10) & np.isfinite(z)
    
    points = np.stack([x[valid_pts], y[valid_pts], z[valid_pts]], axis=1)
    
    return points

def estimate_6dof_pose(points: np.ndarray) -> Dict:
    """Estimate 6-DOF pose using PCA"""
    if points.shape[0] < 10:
        return None
    
    # Compute centroid
    centroid = points.mean(axis=0)
    
    # Center points
    centered = points - centroid
    
    # Compute covariance and eigenvectors
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Ensure right-handed coordinate system
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1
    
    # Compute extents
    projected = centered @ eigvecs
    extents = projected.max(axis=0) - projected.min(axis=0)
    
    return {
        'position': centroid.tolist(),
        'orientation': eigvecs.tolist(),
        'extents': extents.tolist(),
        'eigenvalues': eigvals.tolist()
    }

# =============================================================================
# GRASP PLANNING FOR OCCLUDED/STACKED OBJECTS
# =============================================================================

class OcclusionAwareGraspPlanner:
    """
    Plans grasps considering occlusions and stacking
    """
    
    def plan_grasps(self, detection: Dict, 
                   point_cloud: np.ndarray,
                   stacking_info: Dict) -> List[Dict]:
        """
        Generate grasp proposals considering occlusion and stacking constraints
        """
        grasps = []
        
        # Get object pose
        pose = estimate_6dof_pose(point_cloud)
        if pose is None:
            return []
        
        position = np.array(pose['position'])
        orientation = np.array(pose['orientation'])
        extents = np.array(pose['extents'])
        
        # Check if object is in a stack
        is_stacked = self._is_object_stacked(detection, stacking_info)
        is_top_of_stack = self._is_top_of_stack(detection, stacking_info)
        
        # Generate grasps based on accessibility
        if detection['visible_ratio'] > 0.7 and (not is_stacked or is_top_of_stack):
            # Top-down grasp (preferred for stacked objects)
            grasps.append({
                'type': 'top_down',
                'position': position.tolist(),
                'approach_vector': [0, 0, -1],
                'gripper_width': float(min(extents[0], extents[1]) * 0.8),
                'score': 0.9 * detection['visible_ratio'],
                'feasible': True
            })
        
        # Side grasps only if not stacked and sufficient visibility
        if not is_stacked and detection['visible_ratio'] > 0.5:
            # Side grasp along principal axis
            for axis in range(2):  # Try first two principal axes
                approach = orientation[:, axis]
                grasp_pos = position + approach * (extents[axis] * 0.4)
                
                grasps.append({
                    'type': f'side_axis_{axis}',
                    'position': grasp_pos.tolist(),
                    'approach_vector': approach.tolist(),
                    'gripper_width': float(extents[(axis+1)%3] * 0.7),
                    'score': 0.7 * detection['visible_ratio'],
                    'feasible': not is_stacked
                })
        
        # Penalize grasps for highly occluded objects
        for grasp in grasps:
            grasp['score'] *= (1.0 - detection['occlusion_score'] * 0.5)
        
        # Sort by score
        grasps.sort(key=lambda g: g['score'], reverse=True)
        
        return grasps[:3]  # Return top 3 grasps
    
    def _is_object_stacked(self, detection: Dict, stacking_info: Dict) -> bool:
        """Check if object is part of a stack"""
        det_id = detection.get('id', -1)
        for stack in stacking_info.get('stacks', []):
            if det_id in stack.get('objects', []):
                return True
        return False
    
    def _is_top_of_stack(self, detection: Dict, stacking_info: Dict) -> bool:
        """Check if object is at the top of a stack"""
        det_id = detection.get('id', -1)
        for stack in stacking_info.get('stacks', []):
            objects = stack.get('objects', [])
            if objects and objects[-1] == det_id:
                return True
        return False

# =============================================================================
# MAIN VISION PIPELINE
# =============================================================================

class RoboticVisionPipeline:
    """
    Complete 3D vision pipeline addressing all hackathon requirements:
    - 3D data processing (depth maps, point clouds)
    - Object detection & segmentation with occlusion handling
    - 6-DOF pose estimation for stacked objects
    - Grasp planning for partially occluded objects
    """
    
    def __init__(self):
        logger.info("="*60)
        logger.info("🚀 Initializing Robotic 3D Vision Pipeline")
        logger.info("="*60)
        
        # Initialize components
        self.intrinsics = None
        self.depth_estimator = None
        self.detector = OcclusionAwareDetector(model_size='n')
        self.stacking_analyzer = StackingAnalyzer()
        self.grasp_planner = OcclusionAwareGraspPlanner()
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        # Output directory
        self.output_dir = "vision_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("✅ Pipeline initialized successfully!")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame through complete pipeline
        
        Returns comprehensive analysis including:
        - Detected objects with occlusion analysis
        - Stacking relationships
        - 6-DOF poses
        - Grasp proposals
        """
        start_time = time.time()
        
        # Initialize camera intrinsics on first frame
        if self.intrinsics is None:
            h, w = frame.shape[:2]
            self.intrinsics = CameraIntrinsics.from_fov(w, h, 60.0)
            self.depth_estimator = OcclusionAwareDepthEstimator(self.intrinsics)
            logger.info(f"Camera initialized: {w}x{h}")
        
        # Step 1: Estimate depth with occlusion map
        depth_map, occlusion_map = self.depth_estimator.estimate_depth_with_occlusion_map(frame)
        
        # Step 2: Detect objects with occlusion handling
        detections = self.detector.detect_with_occlusion_handling(frame, depth_map, occlusion_map)
        
        # Step 3: Generate point clouds and estimate poses
        for det in detections:
            if det.get('mask') is not None:
                points = generate_point_cloud_from_depth(depth_map, self.intrinsics, det['mask'])
            else:
                # Use bbox region
                x1, y1, x2, y2 = det['bbox']
                mask = np.zeros(depth_map.shape, dtype=bool)
                mask[y1:y2, x1:x2] = True
                points = generate_point_cloud_from_depth(depth_map, self.intrinsics, mask)
            
            if points.shape[0] > 10:
                pose = estimate_6dof_pose(points)
                det['pose_6dof'] = pose
                det['num_points'] = points.shape[0]
                det['point_cloud'] = points  # Store for grasp planning
            else:
                det['pose_6dof'] = None
                det['num_points'] = 0
                det['point_cloud'] = np.array([])
        
        # Step 4: Analyze stacking relationships
        stacking_info = self.stacking_analyzer.detect_stacks(detections, depth_map)
        
        # Step 5: Plan grasps considering occlusions and stacking
        for i, det in enumerate(detections):
            det['id'] = i  # Assign ID for stacking analysis
            if det['point_cloud'].shape[0] > 0:
                grasps = self.grasp_planner.plan_grasps(det, det['point_cloud'], stacking_info)
                det['grasp_proposals'] = grasps
            else:
                det['grasp_proposals'] = []
            
            # Remove point cloud from dict to avoid serialization issues
            del det['point_cloud']
        
        # Compile results
        process_time = time.time() - start_time
        self.fps_counter.append(process_time)
        fps = 1.0 / np.mean(self.fps_counter) if self.fps_counter else 0
        
        results = {
            'frame_id': self.frame_count,
            'timestamp': time.time(),
            'detections': detections,
            'stacking_analysis': stacking_info,
            'processing_time': process_time,
            'fps': fps,
            'num_objects': len(detections),
            'num_occluded': sum(1 for d in detections if d['occlusion_score'] > 0.3),
            'num_stacked': stacking_info['num_stacked']
        }
        
        # Save results to JSON
        self._save_results(results)
        
        # Create visualization
        viz = self._create_visualization(frame, results, depth_map, occlusion_map)
        
        self.frame_count += 1
        
        return {
            'visualization': viz,
            'results': results
        }
    
    def _create_visualization(self, frame: np.ndarray, results: Dict, 
                            depth_map: np.ndarray, occlusion_map: np.ndarray) -> np.ndarray:
        """Create comprehensive visualization"""
        h, w = frame.shape[:2]
        
        # Create 2x2 grid visualization
        viz = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Top-left: Original with detections
        frame_with_detections = frame.copy()
        for det in results['detections']:
            self._draw_detection(frame_with_detections, det)
        viz[:h, :w] = frame_with_detections
        
        # Top-right: Depth map
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        viz[:h, w:] = depth_colored
        
        # Bottom-left: Occlusion map
        occlusion_colored = cv2.applyColorMap((occlusion_map * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        viz[h:, :w] = occlusion_colored
        
        # Bottom-right: Stacking visualization
        stacking_viz = frame.copy()
        self._draw_stacking_info(stacking_viz, results['stacking_analysis'])
        viz[h:, w:] = stacking_viz
        
        # Add text overlays
        self._add_text_overlays(viz, results)
        
        return viz
    
    def _draw_detection(self, image: np.ndarray, detection: Dict):
        """Draw single detection with occlusion info"""
        x1, y1, x2, y2 = detection['bbox']
        
        # Color based on occlusion level
        if detection['occlusion_score'] > 0.5:
            color = (0, 0, 255)  # Red for highly occluded
        elif detection['occlusion_score'] > 0.3:
            color = (0, 165, 255)  # Orange for partially occluded
        else:
            color = (0, 255, 0)  # Green for visible
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Label with occlusion info
        label = f"{detection['label']} ({detection['visible_ratio']:.0%} visible)"
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw grasp point if available
        if detection.get('grasp_proposals'):
            best_grasp = detection['grasp_proposals'][0]
            if detection.get('pose_6dof'):
                # Project 3D grasp position to 2D (simplified)
                gx = int((x1 + x2) / 2)
                gy = int((y1 + y2) / 2)
                cv2.circle(image, (gx, gy), 5, (255, 255, 0), -1)
                cv2.putText(image, "G", (gx-5, gy-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def _draw_stacking_info(self, image: np.ndarray, stacking_info: Dict):
        """Visualize stacking relationships"""
        for stack in stacking_info['stacks']:
            # Draw connections between stacked objects
            if len(stack['objects']) > 1:
                # Draw stack indicator
                cv2.putText(image, f"Stack: {len(stack['objects'])} objects",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    def _add_text_overlays(self, viz: np.ndarray, results: Dict):
        """Add informative text overlays"""
        texts = [
            f"FPS: {results['fps']:.1f}",
            f"Objects: {results['num_objects']}",
            f"Occluded: {results['num_occluded']}",
            f"Stacked: {results['num_stacked']}",
            f"Max Stack: {results['stacking_analysis']['max_stack_height']}"
        ]
        
        y = 30
        for text in texts:
            cv2.putText(viz, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 25
    
    def _save_results(self, results: Dict):
        """Save results to JSON for robot integration"""
        # Remove non-serializable items
        clean_results = {
            'frame_id': results['frame_id'],
            'timestamp': results['timestamp'],
            'num_objects': results['num_objects'],
            'num_occluded': results['num_occluded'],
            'num_stacked': results['num_stacked'],
            'detections': []
        }
        
        for det in results['detections']:
            clean_det = {
                'label': det['label'],
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'visible_ratio': det['visible_ratio'],
                'occlusion_score': det['occlusion_score'],
                'pose_6dof': det.get('pose_6dof'),
                'grasp_proposals': det.get('grasp_proposals', [])
            }
            clean_results['detections'].append(clean_det)
        
        filename = os.path.join(self.output_dir, f"frame_{results['frame_id']:06d}.json")
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Demonstration of 3D vision system for robotic manipulation
    Addresses all hackathon requirements
    """
    print("\n" + "="*60)
    print("🤖 3D VISION SYSTEM FOR ROBOTIC MANIPULATION")
    print("="*60)
    print("\n✅ CAPABILITIES:")
    print("  • Processes depth maps and generates point clouds")
    print("  • Detects objects even when partially occluded")
    print("  • Identifies stacked objects and relationships")
    print("  • Estimates 6-DOF pose for each object")
    print("  • Plans grasps considering occlusions and stacking")
    print("\n" + "="*60)
    
    # Initialize pipeline
    pipeline = RoboticVisionPipeline()
    
    # Open camera
    print("\n📷 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not found. Please check connection.")
        return
    
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n🎮 CONTROLS:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'p' - Pause/Resume")
    print("  SPACE - Process single frame (when paused)")
    print("\n" + "="*60 + "\n")
    
    paused = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            if not paused:
                # Process frame
                output = pipeline.process_frame(frame)
                viz = output['visualization']
                results = output['results']
                
                # Print summary
                print(f"\rFrame {results['frame_id']}: "
                      f"{results['num_objects']} objects "
                      f"({results['num_occluded']} occluded, "
                      f"{results['num_stacked']} stacked) "
                      f"@ {results['fps']:.1f} FPS", end='')
            else:
                if 'viz' not in locals():
                    viz = frame
                cv2.putText(viz, "PAUSED", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Display
            cv2.imshow('3D Vision System', viz)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n\nShutting down...")
                break
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, viz)
                print(f"\n💾 Saved: {filename}")
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'⏸️ PAUSED' if paused else '▶️ RESUMED'}")
            elif key == ord(' ') and paused:
                output = pipeline.process_frame(frame)
                viz = output['visualization']
                print(f"\nProcessed frame {output['results']['frame_id']}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System shutdown complete")
        print(f"📁 Results saved in: {pipeline.output_dir}/")

if __name__ == "__main__":
    main()