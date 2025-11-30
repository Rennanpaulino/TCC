package com.seutcc.app.adapters;

import android.graphics.Color;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.seutcc.app.R;
import com.seutcc.app.models.Trem;
import java.util.ArrayList;
import java.util.List;

public class TremAdapter extends RecyclerView.Adapter<TremAdapter.TremViewHolder> {

    private List<Trem> listaTrens = new ArrayList<>();
    private OnItemClickListener listener;

    public interface OnItemClickListener {
        void onItemClick(Trem trem);
    }

    public TremAdapter(OnItemClickListener listener) {
        this.listener = listener;
    }

    public void atualizarLista(List<Trem> novosTrens) {
        this.listaTrens = novosTrens;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public TremViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_trem, parent, false);
        return new TremViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull TremViewHolder holder, int position) {
        Trem trem = listaTrens.get(position);

        // 1. Dados Básicos
        holder.txtId.setText(trem.getId());
        holder.txtLotacao.setText("Lotação: " + trem.getLotacao() + " pessoas");

        // 2. Cores da Lotação
        if (trem.getLotacao() < 10) {
            holder.viewStatus.setBackgroundColor(Color.parseColor("#4CAF50")); // Verde
        } else if (trem.getLotacao() < 30) {
            holder.viewStatus.setBackgroundColor(Color.parseColor("#FFC107")); // Amarelo
        } else {
            holder.viewStatus.setBackgroundColor(Color.parseColor("#F44336")); // Vermelho
        }

        // 3. NOVO: Exibir ETA (Tempo de Chegada)
        // O Backend manda em minutos e segundos
        String textoEta = String.format("Chegada: %02d min", trem.getEtaMinutos());
        holder.txtEta.setText(textoEta);

        // 4. NOVO: Aviso de GPS Estimado
        if (trem.isEstimado()) {
            holder.txtAviso.setVisibility(View.VISIBLE);
            holder.txtEta.setTextColor(Color.GRAY); // Deixa o ETA cinza para indicar incerteza
        } else {
            holder.txtAviso.setVisibility(View.GONE);
            holder.txtEta.setTextColor(Color.parseColor("#0055FF")); // Azul normal
        }

        // Clique para abrir detalhes
        holder.itemView.setOnClickListener(v -> listener.onItemClick(trem));
    }

    @Override
    public int getItemCount() {
        return listaTrens.size();
    }

    static class TremViewHolder extends RecyclerView.ViewHolder {
        TextView txtId, txtLotacao, txtEta, txtAviso;
        View viewStatus;

        public TremViewHolder(@NonNull View itemView) {
            super(itemView);
            txtId = itemView.findViewById(R.id.txtIdTrem);
            txtLotacao = itemView.findViewById(R.id.txtLotacao);
            viewStatus = itemView.findViewById(R.id.viewStatus);

            // Novos campos mapeados
            txtEta = itemView.findViewById(R.id.txtEtaLista);
            txtAviso = itemView.findViewById(R.id.txtAvisoLista);
        }
    }
}