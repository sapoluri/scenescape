// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";
import { activateSceneTab, tabForSheetAction } from "../lib/sceneTab";
import { workspaceEditingAllowed } from "../lib/editMode";
import {
  clearSheetQuery,
  parseSheetQuery,
  type SheetAction,
  type SheetQuery,
} from "../lib/sheetQuery";

/**
 * Opens a sheet from ?ss=&id= on mount, then strips the query so refresh
 * does not re-open forever.
 */
export function useSheetFromQuery() {
  const [sheet, setSheet] = useState<SheetQuery>(() => parseSheetQuery());

  useEffect(() => {
    const initial = parseSheetQuery();
    if (initial.action) {
      setSheet(initial);
      clearSheetQuery();
    }
  }, []);

  const open = useCallback(
    (action: Exclude<SheetAction, null>, id: string | null = null) => {
      if (!workspaceEditingAllowed()) return;
      setSheet({ action, id });
      const tab = tabForSheetAction(action);
      if (tab) activateSceneTab(tab);
    },
    [],
  );

  const close = useCallback(() => {
    setSheet({ action: null, id: null });
  }, []);

  return { sheet, open, close, setSheet };
}
