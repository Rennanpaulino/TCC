package com.seutcc.app.adapters;

import android.graphics.Color;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.seutcc.app.R;
import com.seutcc.app.models.Linha;
import java.util.List;

public class LinhaAdapter extends RecyclerView.Adapter<LinhaAdapter.ViewHolder> {

    private List<Linha> lista;
    private OnClick listener;

    public interface OnClick { void onClick(Linha linha); }

    public LinhaAdapter(List<Linha> lista, OnClick listener) {
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
        Linha linha = lista.get(position);
        holder.txtNome.setText(linha.getNome());
        holder.txtNome.setTextSize(18);
        holder.txtNome.setPadding(32, 32, 32, 32);

        // Pinta o texto com a cor da linha (Ex: Amarela, Esmeralda)
        try {
            holder.txtNome.setTextColor(Color.parseColor(linha.getCorHex()));
        } catch (Exception e) {
            holder.txtNome.setTextColor(Color.BLACK);
        }

        holder.itemView.setOnClickListener(v -> listener.onClick(linha));
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