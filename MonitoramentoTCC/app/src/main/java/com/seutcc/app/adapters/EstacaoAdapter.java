package com.seutcc.app.adapters;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.seutcc.app.models.Estacao;
import java.util.List;

public class EstacaoAdapter extends RecyclerView.Adapter<EstacaoAdapter.ViewHolder> {

    private List<Estacao> lista;
    private OnClick listener;

    public interface OnClick { void onClick(Estacao estacao); }

    public EstacaoAdapter(List<Estacao> lista, OnClick listener) {
        this.lista = lista;
        this.listener = listener;
    }

    @NonNull @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(android.R.layout.simple_list_item_1, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Estacao estacao = lista.get(position);
        holder.txtNome.setText(estacao.getNome());
        holder.itemView.setOnClickListener(v -> listener.onClick(estacao));
    }

    @Override public int getItemCount() { return lista.size(); }

    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView txtNome;
        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            txtNome = itemView.findViewById(android.R.id.text1);
        }
    }
}