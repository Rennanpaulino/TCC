package com.seutcc.app.models;

public class Report {
    private String usuario;
    private String trem_id;
    private String tipo_problema;
    private String descricao;

    public Report(String usuario, String trem_id, String tipo_problema, String descricao) {
        this.usuario = usuario;
        this.trem_id = trem_id;
        this.tipo_problema = tipo_problema;
        this.descricao = descricao;
    }
}