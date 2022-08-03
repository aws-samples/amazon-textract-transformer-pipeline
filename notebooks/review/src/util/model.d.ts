// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
/**
 * Type declarations for expected A2I task input data for this task
 */

/**
 * Detected instance of a given entity/field type in document text
 */
export interface Detection {
  Blocks: string[];
  BoundingBox: { Height: number; Left: number; Top: number; Width: number };
  ClassId: number;
  ClassName: string;
  Confidence: number;
  PageNum: number;
  Text: string;
}

/**
 * Common interface for single- or multi-value entity/field results.
 */
interface ModelResultFieldBase {
  ClassId: number;
  Confidence: number;
  NumDetectedValues: number;
  NumDetections: number;
  Optional?: boolean;
  SortOrder: number;
}

/**
 * Overall detection result for a given entity/field type.
 */
export interface ModelResultSingleField extends ModelResultFieldBase {
  Detections: Detection[];
  Value: string;
}

/**
 * Overall detection result for a given entity/field type.
 */
export interface ModelResultMultiField extends ModelResultFieldBase {
  Values: Array<{
    Confidence: number;
    Detections: Detection[];
    Value: string;
  }>;
}

/**
 * Overall model result across entity/field types.
 */
export interface ModelResult {
  Confidence: number;
  Fields: {
    [FieldName: string]: ModelResultSingleField | ModelResultMultiField;
  };
}
