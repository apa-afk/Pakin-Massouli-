


def excel_fix(df):
    # header is on row 1
    df.columns = df.iloc[1].astype(str)

    # clean year column names (keep first column)
    cols = df.columns.tolist()
    cols[1:] = [str(int(float(c.replace("*", "").strip()))) for c in cols[1:]]
    df.columns = cols

    # keep data rows
    df = df.iloc[2:].reset_index(drop=True)
    df = df.iloc[:17].copy()

    # rename first column
    df = df.rename(columns={df.columns[0]: "region"})

    # normalize region names
    df["region"] = (
        df["region"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            "PROVENCE-ALPES-COTE -D'AZUR": "PROVENCE-ALPES-CÔTE D'AZUR",
            "AUVERGNE RHONE ALPES": "AUVERGNE-RHÔNE-ALPES",
            "CENTRE VAL DE LOIRE": "CENTRE-VAL DE LOIRE",
            "BOURGOGNE FRANCHE COMTE": "BOURGOGNE-FRANCHE-COMTÉ",
            "PAYS DE LA LOIRE": "PAYS DE LA LOIRE",
            "ILE-DE-FRANCE": "ÎLE-DE-FRANCE",
            "NOMBRE TOTAL": "TOTAL",
            "MONTANT TOTAL": "TOTAL",
        })
    )

    # region -> INSEE reg code (same as df_conso.reg)
    region_to_reg = {
        "ÎLE-DE-FRANCE": 11,
        "CENTRE-VAL DE LOIRE": 24,
        "BOURGOGNE-FRANCHE-COMTÉ": 27,
        "NORMANDIE": 28,
        "HAUTS DE FRANCE": 32,
        "GRAND EST": 44,
        "PAYS DE LA LOIRE": 52,
        "BRETAGNE": 53,
        "NOUVELLE AQUITAINE": 75,
        "OCCITANIE": 76,
        "AUVERGNE-RHÔNE-ALPES": 84,
        "PROVENCE-ALPES-CÔTE D'AZUR": 93,
        "CORSE": 94,
        "OUTRE-MER": 1,
        "RÉSIDENTS ÉTRANGERS": 99,
        "TOTAL": 0,
    }

    df["region_id"] = df["region"].map(region_to_reg)

    # drop unmapped rows
    df = df[df["region_id"].notna()].copy()
    df["region_id"] = df["region_id"].astype(int)

    # reorder columns
    cols = ["region_id", "region"] + [c for c in df.columns if c not in ["region_id", "region"]]
    return df[cols]





def compute_richesse(df1_fixed, df2_fixed):
    # copy structure
    richesse_df = df2_fixed.copy()

    # numeric part: all columns except region_id and region
    value_cols = richesse_df.columns[2:]

    # element-wise division
    richesse_df[value_cols] = df2_fixed[value_cols] / df1_fixed[value_cols]

    return richesse_df

